# Kaggle-rehab
利用 Kaggle 上的康复数据集（以 “Stroke Rehabilitation Prediction” 为例）进行基础分析的 Python 示例
以下是一个利用Kaggle上的康复数据集（以“Stroke Rehabilitation Prediction”为例）进行基础分析的Python示例。假设你已经下载了数据集，这里我们将进行数据加载、数据探查、数据可视化等基础操作。

### 1. 环境准备
首先确保你已经安装了必要的库，如`pandas`、`numpy`、`matplotlib`和`seaborn`。可以使用以下命令进行安装：
```bash
pip install pandas numpy matplotlib seaborn
```

### 2. 代码实现

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载
# 请将 'stroke_rehabilitation_data.csv' 替换为你实际的数据集文件名
data = pd.read_csv('stroke_rehabilitation_data.csv')

# 2. 数据基本信息查看
print("数据基本信息：")
data.info()

# 3. 查看数据集行数和列数
rows, columns = data.shape

if rows < 1000:
    # 小数据集（行数少于1000）查看全量数据信息
    print("数据全部内容信息：")
    print(data.to_csv(sep='\t', na_rep='nan'))
else:
    # 大数据集查看数据前几行信息
    print("数据前几行内容信息：")
    print(data.head().to_csv(sep='\t', na_rep='nan'))

# 4. 查看数据集行数和列数
rows, columns = data.shape

# 5. 数据描述性统计
# 对于数值型特征
numerical_features = data.select_dtypes(include=[np.number])
print("\n数值型特征的描述性统计：")
print(numerical_features.describe())

# 对于分类型特征
categorical_features = data.select_dtypes(include=['object'])
if not categorical_features.empty:
    print("\n分类型特征的描述性统计：")
    print(categorical_features.describe())

# 6. 缺失值分析
missing_values = data.isnull().sum()
print("\n各列缺失值数量：")
print(missing_values)

# 7. 可视化缺失值分布
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# 8. 数值型特征分布可视化
for col in numerical_features.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 9. 分类型特征分布可视化
if not categorical_features.empty:
    for col in categorical_features.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=data)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.show()

# 10. 相关性分析（仅针对数值型特征）
if len(numerical_features.columns) > 1:
    correlation_matrix = numerical_features.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

```

### 3. 代码解释
1. **数据加载**：使用`pandas`的`read_csv`函数读取CSV格式的数据集。
2. **数据基本信息查看**：通过`info()`方法查看数据的基本信息，包括列名、数据类型、非空值数量等。
3. **数据内容查看**：根据数据集行数的多少，决定查看全量数据信息还是前几行数据信息。
4. **描述性统计**：分别对数值型特征和分类型特征进行描述性统计分析，数值型特征使用`describe()`方法，分类型特征也使用`describe()`方法查看其统计信息。
5. **缺失值分析**：计算每列的缺失值数量，并使用热力图可视化缺失值的分布情况。
6. **特征分布可视化**：对于数值型特征，使用直方图和核密度估计图展示其分布；对于分类型特征，使用计数图展示其分布。
7. **相关性分析**：计算数值型特征之间的相关性矩阵，并使用热力图进行可视化。

### 4. 注意事项
- 请将代码中的`'stroke_rehabilitation_data.csv'`替换为你实际下载的数据集文件名。
- 如果数据集中存在日期类型的特征，需要先进行日期类型的转换，再进行相关分析。
- 以上代码只是基础分析，你可以根据具体的业务需求和数据集特点进行更深入的分析。
