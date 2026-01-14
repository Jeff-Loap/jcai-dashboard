import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
csv_path = r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features_with_regime.csv"
df = pd.read_csv(csv_path)

# 选取关键变量
cols = [
    "y_ret_k",
    "y_vol_k",
    "ret_1_L1_z120",
    "range_hl_L1_z120",
    "macd_signal_L1_z120",
    "rv_20_L1_z120",
]
eda_df = df[cols].dropna()

# 1) y_ret_k 分布
plt.figure()
plt.hist(eda_df["y_ret_k"], bins=40)
plt.title("Distribution of y_ret_k")
plt.xlabel("y_ret_k")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2) y_vol_k 分布
plt.figure()
plt.hist(eda_df["y_vol_k"], bins=40)
plt.title("Distribution of y_vol_k")
plt.xlabel("y_vol_k")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3) 关键因子 vs y_ret_k 散点
for col in ["ret_1_L1_z120", "range_hl_L1_z120", "macd_signal_L1_z120"]:
    plt.figure()
    plt.scatter(eda_df[col], eda_df["y_ret_k"], s=10)
    plt.title(f"{col} vs y_ret_k")
    plt.xlabel(col)
    plt.ylabel("y_ret_k")
    plt.tight_layout()
    plt.show()

# 4) 相关矩阵热力图
corr = eda_df.corr()
plt.figure()
plt.imshow(corr, interpolation="nearest")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Matrix (Selected Features)")
plt.tight_layout()
plt.show()
