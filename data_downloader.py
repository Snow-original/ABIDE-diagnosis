from nilearn import datasets
import pandas as pd

# 1. 下载 ABIDE 预处理数据
# pipeline: 'cpac', 'ccs', 'niak', 'dparsf' (常用 cpac)
# derivatives: 'rois_aal', 'rois_ho' (常用的感兴趣区坐标)
data = datasets.fetch_abide_pcp(
    n_subjects=None,          # 先试着下载100个受试者，设为None下载全部(1112个)
    pipeline='cpac',         # 预处理流水线
    derivatives='rois_aal',  # 下载AAL图谱定义的ROI时间序列
    quality_checked=True     # 只保留通过质量检测的数据（推荐）
)

# 2. 获取表型数据 (Phenotypic Data)
# 这包含了年龄、性别、诊断结果(DX_GROUP)等
pheno_df = pd.DataFrame(data.phenotypic)
print("表型数据预览：")
print(pheno_df.head())

# 3. 获取 fMRI 数据路径 (ROI Time Series)
# 这是一个列表，每个元素对应一个受试者的 .1D 文件路径
fmri_paths = data.rois_aal
print(f"\n成功获取 {len(fmri_paths)} 个受试者的 fMRI ROI 数据路径。")

# 保存表型数据到本地方便后续分析
pheno_df.to_csv("abide_phenotypic_data.csv", index=False)