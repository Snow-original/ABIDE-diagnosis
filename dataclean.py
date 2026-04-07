import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 1. 明确论文保留的 12 个核心特征 + 两个必要 ID/标签
selected_features = [
    'ADOS_SOCIAL', 'ADI_R_ONSET_TOTAL_D', 'ADI_RRB_TOTAL_C', 
    'ADI_R_VERBAL_TOTAL_BV', 'ADOS_STEREO_BEHAV', 'ADOS_TOTAL', 
    'VIQ', 'FIQ', 'ADI_R_SOCIAL_TOTAL_A', 'SEX', 'SITE_ID', 'AGE_AT_SCAN'
]
metadata = ['SUB_ID', 'DX_GROUP']

# 2. 读取并清洗
df = pd.read_csv("abide_phenotypic_data.csv") # 请确保指向你最全的那个表型原始文件
df.replace([-9999, '-9999'], np.nan, inplace=True)

# 3. 提取子集
# 注意：有些特征名在不同版本的 PCP 中可能微调（如 ADOS_TOTAL_8 改为 ADOS_TOTAL），请根据你的 CSV 确认
present_features = [f for f in selected_features if f in df.columns]
final_df = df[metadata + present_features].copy()

# 4. KNN 填补预处理
impute_cols = present_features

# 如果 SITE_ID 是字符串，先转成数字（必须在 data_to_impute 赋值之前完成）
if final_df['SITE_ID'].dtype == object:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    final_df['SITE_ID'] = le.fit_transform(final_df['SITE_ID'].astype(str))

data_to_impute = final_df[impute_cols].copy()

# 5. 执行论文所述的 KNN 填补
# 论文通常使用 k=5 或 k=10
imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputed_data = imputer.fit_transform(data_to_impute)

# 6. 将填补后的数据放回原表
final_df[impute_cols] = imputed_data

# 7. 导出最终可以直接用于模型训练的表型特征
final_df.to_csv("final_pheno_for_fusion.csv", index=False)
print("✅ 最终多模态特征表已生成，样本量：", len(final_df))
