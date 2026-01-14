import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# 补充引用 roc_curve 和 auc 用于画图
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

pos_file = "../positive.fasta"
neg_file = "../negative.fasta"
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def get_aac_feature(seq):
    clean_seq = [aa for aa in seq if aa in amino_acids]
    length = len(clean_seq)
    if length == 0: return [0] * 20
    count_dict = {aa: 0 for aa in amino_acids}
    for aa in clean_seq: count_dict[aa] += 1
    return [count_dict[aa] / length for aa in amino_acids]

def get_cksaap_feature(seq, gap=0):
    aa_pairs = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    pair_dict = {pair: i for i, pair in enumerate(aa_pairs)}
    feature = [0] * 400
    length = len(seq)
    if length <= gap + 1: return feature
    total_pairs = 0
    for i in range(length - gap - 1):
        pair = seq[i] + seq[i + gap + 1]
        if pair in pair_dict:
            feature[pair_dict[pair]] += 1
            total_pairs += 1
    if total_pairs > 0:
        feature = [count / total_pairs for count in feature]
    return feature


def get_physio_feature(seq):
    # 疏水性值 (Hydrophobicity Index)
    hydrophobicity_map = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    # 电荷值 (Charge)
    charge_map = {
        'K': 1, 'R': 1, 'H': 0.1,  # 正电荷
        'D': -1, 'E': -1,  # 负电荷
    }
    total_hydro = 0
    total_charge = 0
    valid_len = 0
    for aa in seq:
        if aa in amino_acids:
            total_hydro += hydrophobicity_map.get(aa, 0)
            total_charge += charge_map.get(aa, 0)
            valid_len += 1
    if valid_len == 0:
        return [0, 0]
    return [total_hydro / valid_len, total_charge / valid_len]


print("正在读取 FASTA 文件...")
X = []
y = []

def process_file(filename, label):
    count = 0
    for record in SeqIO.parse(filename, "fasta"):
        seq_str = str(record.seq).upper()
        # 特征提取
        feat_aac = get_aac_feature(seq_str)  # 20维
        feat_k0 = get_cksaap_feature(seq_str, gap=0)  # 400维
        feat_k1 = get_cksaap_feature(seq_str, gap=1)  # 400维
        feat_phy = get_physio_feature(seq_str)  # 2维

        # 特征融合
        final_feat = list(feat_aac) + list(feat_k0) + list(feat_k1) + list(feat_phy)

        X.append(final_feat)
        y.append(label)
        count += 1
    return count

c1 = process_file(pos_file, 1)
c0 = process_file(neg_file, 0)
print(f"读取完毕: 正样本 {c1} 条, 负样本 {c0} 条")

X = np.array(X)
y = np.array(y)
print(f"原始数据维度: {X.shape}")


print("\n⚖️ 正在进行数据标准化 (StandardScaler)...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"标准化完成！均值: {np.mean(X):.2f}, 方差: {np.std(X):.2f}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("\n🔍 正在进行特征筛选...")

# 生成原始特征名字
orig_feat_names = list(amino_acids)
aa_pairs = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
orig_feat_names += [f"{pair}_gap0" for pair in aa_pairs]
orig_feat_names += [f"{pair}_gap1" for pair in aa_pairs]
orig_feat_names += ["Avg_Hydrophobicity", "Avg_Charge"]

print(f"预期特征总数: {len(orig_feat_names)} (应为 822)")

# 粗筛
selector_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
selector_model.fit(X_train, y_train)

selection = SelectFromModel(selector_model, threshold="1.2*mean", prefit=True)
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)

# 获取被选中的特征名字
selected_indices = selection.get_support(indices=True)
selected_feat_names = [orig_feat_names[i] for i in selected_indices]

print(f"✅ 筛选完成！维度变化: {X_train.shape[1]} -> {select_X_train.shape[1]}")


print("\n🤝 正在组建模型联盟 (RandomForest + XGBoost)...")

rf_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

xgb_best = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric='logloss'
)

ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_best)],
    voting='soft',
    n_jobs=-1
)

ensemble_model.fit(select_X_train, y_train)

preds = ensemble_model.predict(select_X_test)
probs = ensemble_model.predict_proba(select_X_test)[:, 1]

acc = accuracy_score(y_test, preds)
auc_score = roc_auc_score(y_test, probs)

print("-" * 30)
print(f"🚀 [融合模型] 最终测试集准确率: {acc:.4f}")
print(f"🔥 [融合模型] 最终测试集 AUC   : {auc_score:.4f}")
print("-" * 30)

print("\n详细分类报告:")
print(classification_report(y_test, preds))


cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Ensemble)')
plt.show()


print("\n🔍 正在分析融合模型的特征重要性...")

rf_fitted = ensemble_model.estimators_[0]
xgb_fitted = ensemble_model.estimators_[1]

rf_imp = rf_fitted.feature_importances_
xgb_imp = xgb_fitted.feature_importances_
avg_imp = (rf_imp + xgb_imp) / 2

indices = np.argsort(avg_imp)[::-1]

print("-" * 30)
print("🔥 [融合模型] 认为最重要的 Top 15 特征:")
print("-" * 30)

for f in range(min(15, len(indices))):
    idx = indices[f]
    score = avg_imp[idx]
    name = selected_feat_names[idx]
    print(f"{f + 1:2d}. {name:<20} (权重: {score:.4f})")


print("\n🧪 物理化学特征 (Physicochemical) 表现如何？")
phy_features = ["Avg_Hydrophobicity", "Avg_Charge"]

for phy_name in phy_features:
    if phy_name in selected_feat_names:
        real_idx = selected_feat_names.index(phy_name)
        real_score = avg_imp[real_idx]
        rank = np.where(indices == real_idx)[0][0] + 1
        print(f"  -> {phy_name}: 排名第 {rank} / {len(selected_feat_names)}, 权重: {real_score:.4f}")
    else:
        print(f"  -> {phy_name}: ❌ 在特征筛选阶段被剔除了")


print("\n🎨 正在绘制最终的 ROC 对比图 (Figure 3)...")

# 单独训练 RF 和 XGB 以便画对比线
rf_model.fit(select_X_train, y_train)
y_prob_rf = rf_model.predict_proba(select_X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

xgb_best.fit(select_X_train, y_train)
y_prob_xgb = xgb_best.predict_proba(select_X_test)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# 融合模型 (已经训练过，直接预测)
y_prob_ens = ensemble_model.predict_proba(select_X_test)[:, 1]
fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)
roc_auc_ens = auc(fpr_ens, tpr_ens)

plt.figure(figsize=(8, 6), dpi=150)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, alpha=0.6, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, alpha=0.6, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
plt.plot(fpr_ens, tpr_ens, color='red', lw=3, label=f'Proposed Ensemble (AUC = {roc_auc_ens:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.show()


print("\n📊 正在生成 Figure 2 (混淆矩阵对比图)...")


# 定义一个画图函数，方便重复使用
def plot_cm(model, X_test, y_test, title, filename):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title, fontsize=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"✅ 已保存: {filename}")


# 1. 绘制 (a) Random Forest
plot_cm(rf_model, select_X_test, y_test,
        '(a) Random Forest', '../Figure2_a_RandomForest_CM.png')

# 2. 绘制 (b) XGBoost
plot_cm(xgb_best, select_X_test, y_test,
        '(b) XGBoost', '../Figure2_b_XGBoost_CM.png')

# 3. 绘制 (c) Ensemble (Proposed)
plot_cm(ensemble_model, select_X_test, y_test,
        '(c) Ensemble Model', '../Figure2_c_Ensemble_CM.png')

