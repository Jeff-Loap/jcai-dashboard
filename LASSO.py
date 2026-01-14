import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from numpy import logspace
import numpy as np
import matplotlib.pyplot as plt

# ========= 1. 读入数据 =========
csv_path = r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features_with_regime.csv"
df = pd.read_csv(csv_path)

# ========= 1.1 读入“LASSO 文档”里的特征名 =========
# 这份 txt 是你大脚本 lasso_select() 导出的特征列表
lasso_feat_path = r"D:\PythonFile\JCAI\data_ak\outputs\lasso_selected_features.txt"
with open(lasso_feat_path, "r", encoding="utf-8") as f:
    selected_from_doc = [line.strip() for line in f if line.strip()]

print("文档里 LASSO 选出的特征数量:", len(selected_from_doc))
print("文档前几个特征示例:", selected_from_doc[:5])

# ========= 2. 配置：目标列 & 不参与建模的列 =========
target_col = "y_ret_k"   # 用未来收益率作为回归目标

drop_cols = [
    "ticker",
    "trade_date",
    "regime_id",
    "regime_risk_label",
    "y_up_k",
    "y_vol_k",
]

# ========= 3. 拆分特征和标签 =========
df_feat = df.drop(columns=[target_col] + drop_cols, errors="ignore")

# 根据 LASSO 文档里的特征名做交集
use_cols = [c for c in selected_from_doc if c in df_feat.columns]
print("最终用于建模的特征数量:", len(use_cols))

X = df_feat[use_cols].select_dtypes(include=["number"])
y = df[target_col]

print("特征维度:", X.shape)

# ========= 4. 划分训练集 / 测试集 =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========= 5. 先用 LassoCV 找一个“参考 alpha” =========
lasso_cv_pipe = make_pipeline(
    StandardScaler(),
    LassoCV(
        alphas=logspace(-5, -2, 50),  # 1e-5 ~ 1e-2 之间做交叉验证
        cv=5,
        random_state=42,
        n_jobs=-1,
        max_iter=50000,
    ),
)

lasso_cv_pipe.fit(X_train, y_train)
lasso_cv = lasso_cv_pipe.named_steps["lassocv"]
alpha_cv = lasso_cv.alpha_
print("交叉验证选择到的最佳 alpha:", alpha_cv)

# ========= 6. 用更小的 alpha 训练真正用于评估/再筛选的 Lasso =========
alpha_manual = alpha_cv / 10.0 if alpha_cv > 0 else 1e-5

lasso_pipe = make_pipeline(
    StandardScaler(),
    Lasso(
        alpha=alpha_manual,
        max_iter=50000,
        random_state=42,
    ),
)

lasso_pipe.fit(X_train, y_train)
lasso = lasso_pipe.named_steps["lasso"]
print("用于特征筛选的 alpha:", lasso.alpha)

# ========= 7. 查看特征选择结果 =========
coef = lasso.coef_
feature_names = np.array(use_cols)

selected_mask = coef != 0
selected_features = feature_names[selected_mask]

print("\n被 LASSO 再次选中的特征数量:", selected_mask.sum())
print("被选中特征列表：")
for f in selected_features:
    print(f)

# 系数 DataFrame（给排序和画图用）
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coef
})
coef_df = coef_df[coef_df["coef"] != 0]
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\n按重要性排序的特征（非零系数）：")
print(coef_df[["feature", "coef"]])

# ========= 8. 简单评估一下模型效果 + 可视化 =========
y_pred_train = lasso_pipe.predict(X_train)
y_pred_test = lasso_pipe.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)

print("\n训练集 R2:", r2_train)
print("测试集  R2:", r2_test)
print("测试集  MSE:", mse_test)

# 8.1 LASSO 系数条形图（只画非零系数）
if not coef_df.empty:
    plt.figure()
    coef_df_plot = coef_df.sort_values("abs_coef", ascending=True)
    plt.barh(coef_df_plot["feature"], coef_df_plot["coef"])
    plt.title("LASSO Coefficients (Selected Features)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()

# 8.2 测试集预测值 vs 实际值
plt.figure()
plt.scatter(y_test, y_pred_test, s=10)
plt.title("Predicted vs Actual y_ret_k (Test Set)")
plt.xlabel("Actual y_ret_k")
plt.ylabel("Predicted y_ret_k")
plt.tight_layout()
plt.show()

# 8.3 测试集残差分布
resid = y_test - y_pred_test
plt.figure()
plt.hist(resid, bins=40)
plt.title("Residuals Distribution (Test Set)")
plt.xlabel("Residual = actual - predicted")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ========= 9. 导出只包含“再筛选后特征 + 目标”的数据集（可选） =========
out_cols = list(selected_features) + [target_col]
df_selected = df[out_cols]

save_path = r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features_lasso_selected.csv"
df_selected.to_csv(save_path, index=False)
print("\n已保存筛选后数据到：", save_path)
