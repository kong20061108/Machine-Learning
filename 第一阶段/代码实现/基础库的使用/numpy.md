# NumPy 在机器学习中的常用函数笔记

## 1. 概述

- **NumPy 简介**：NumPy 是 Python 用于科学计算的核心库，提供高效的多维数组 `ndarray` 和数学函数。在机器学习中，它用于数据处理、特征工程、模型计算（如矩阵运算和梯度下降）。吴恩达的《Machine Learning》课程中，NumPy 用于实现线性回归、逻辑回归等算法的核心计算。
- **导入**：`import numpy as np`
- **优势**：向量化操作避免循环，提高效率；支持广播机制，简化不同形状数组运算。
- **安装**：`pip install numpy`（通常已安装）。

## 2. 创建数组的函数

这些函数用于生成数据矩阵（如特征 $X$ 或标签 $y$），在机器学习中常用于初始化数据集或参数。

- `np.array(object, dtype=None)`：从列表或元组创建数组。  
  用处：将 Python 列表转换为高效数组，例如创建特征矩阵 $X$ 或标签向量 $y$。  
  示例：`np.array([[1, 2], [3, 4]])` → 2D 数组。

- `np.zeros(shape, dtype=float)`：创建全零数组。  
  用处：初始化参数 $\theta$ 或梯度向量，避免随机初始值。  
  示例：`np.zeros((3, 1))` → 3x1 零矩阵。

- `np.ones(shape, dtype=None)`：创建全一数组。  
  用处：添加偏置项（如线性回归的 $X$ 加一列全 1），或初始化权重。  
  示例：`np.ones((2, 2))` → 全 1 矩阵。

- `np.full(shape, fill_value)`：创建填充特定值的数组。  
  用处：初始化常量矩阵，如学习率矩阵。  
  示例：`np.full((2, 3), 5)` → 全 5 数组。

- `np.arange(start, stop, step)`：创建等差序列数组。  
  用处：生成索引或模拟数据序列，如时间序列特征。  
  示例：`np.arange(0, 10, 2)` → [0, 2, 4, 6, 8]。

- `np.linspace(start, stop, num)`：创建等间隔数组。  
  用处：生成均匀分布的测试数据或学习率衰减序列。  
  示例：`np.linspace(0, 1, 5)` → [0, 0.25, 0.5, 0.75, 1]。

- `np.eye(N)` 或 `np.identity(N)`：创建单位矩阵。  
  用处：初始化权重矩阵，或线性代数计算（如正则化）。  
  示例：`np.eye(3)` → 3x3 单位矩阵。

## 3. 数组属性函数

这些用于检查数据形状和类型，在调试机器学习模型时非常有用（如确保 $X$ 是 m x n 矩阵）。

- `arr.shape`：返回数组形状（元组）。  
  用处：检查特征矩阵维度，例如 $X.shape$ 应为 (样本数, 特征数)。

- `arr.ndim`：返回数组维度数。  
  用处：验证数据是否为预期维度（如 2D 矩阵用于回归）。

- `arr.size`：返回元素总数。  
  用处：计算样本数 $m = X.size / X.shape[1]$。

- `arr.dtype`：返回元素数据类型。  
  用处：确保数据类型一致（如 float64 用于浮点计算，避免整数溢出）。

## 4. 基本操作函数

这些支持向量化运算，在机器学习中避免 for 循环，提高速度（如计算预测 $h = X \theta$）。

- `+`, `-`, `*`, `/`：元素级加减乘除。  
  用处：批量数据处理，如特征归一化 $x' = (x - \mu) / \sigma$。

- `np.dot(a, b)` 或 `a @ b`：点积（矩阵乘法）。  
  用处：计算线性回归预测 $h = X \theta$，或梯度 $\nabla = X^T (h - y)$。

- `np.transpose(arr)` 或 `arr.T`：转置数组。  
  用处：矩阵转置，如正规方程 $(X^T X)^{-1}$。

- `np.broadcast_to(arr, shape)`：显式广播数组到新形状。  
  用处：处理不同维度数据，如将向量广播到矩阵用于批量运算。

## 5. 统计函数

在机器学习中，用于数据探索、特征缩放和损失计算（如均方误差）。

- `np.mean(arr, axis=None)`：计算均值。  
  用处：特征缩放的 $\mu$，或损失函数计算 $J(\theta) = \frac{1}{2m} \sum (h - y)^2$ 中的平均。

- `np.std(arr, axis=None)`：计算标准差。  
  用处：特征缩放的 $\sigma$，或评估模型方差。

- `np.var(arr, axis=None)`：计算方差。  
  用处：计算数据分散度，或正则化项。

- `np.sum(arr, axis=None)`：求和。  
  用处：损失函数求和 $\sum (h - y)^2$，或梯度计算。

- `np.max(arr, axis=None)` / `np.min(arr, axis=None)`：最大/最小值。  
  用处：归一化缩放的 $x_{\max}$ / $x_{\min}$，或找到异常值。

- `np.argmin(arr)` / `np.argmax(arr)`：最小/最大值索引。  
  用处：分类中找到最大概率类，或调试梯度下降最小值。

## 6. 线性代数函数（np.linalg 模块）

在机器学习中，用于正规方程、PCA 等。

- `np.linalg.inv(arr)`：计算矩阵逆。  
  用处：正规方程 $\theta = (X^T X)^{-1} X^T y$。

- `np.linalg.det(arr)`：计算行列式。  
  用处：检查矩阵是否可逆（行列式 ≠ 0）。

- `np.linalg.eig(arr)`：计算特征值和特征向量。  
  用处：PCA 降维，计算协方差矩阵的特征值。

- `np.linalg.svd(arr)`：奇异值分解。  
  用处：降维或矩阵分解，如在推荐系统中。

## 7. 随机函数（np.random 模块）

用于初始化参数、生成模拟数据或数据增强。

- `np.random.rand(shape)`：生成 [0, 1) 均匀随机数组。  
  用处：初始化权重 $\theta$，避免梯度下降卡在局部最小。

- `np.random.randn(shape)`：生成标准正态分布随机数组。  
  用处：初始化神经网络权重，或生成噪声数据测试模型鲁棒性。

- `np.random.randint(low, high, size)`：生成整数随机数组。  
  用处：随机采样索引，或生成分类标签。

- `np.random.seed(seed)`：设置随机种子。  
  用处：确保实验可复现，如固定随机初始化。

- `np.random.shuffle(arr)`：随机打乱数组。  
  用处：随机化训练数据，防止模型过拟合顺序。

## 8. 形状和维度函数

用于数据重塑，在机器学习中常见于预处理（如扁平化图像）。

- `np.reshape(arr, newshape)`：改变数组形状。  
  用处：将一维数据转为矩阵，如图像展平为向量。

- `np.flatten(arr)` 或 `arr.ravel()`：展平为 1D 数组。  
  用处：将多维特征转为向量输入模型。

- `np.concatenate((arr1, arr2), axis=0)`：沿轴连接数组。  
  用处：合并数据集或添加特征。

- `np.vstack(arrs)` / `np.hstack(arrs)`：垂直/水平堆叠。  
  用处：添加偏置列，如 $X_b = [1, X]$。

## 9. 索引和掩码函数

用于数据选择和过滤，在特征工程中关键。

- `arr[mask]`：布尔掩码索引。  
  用处：过滤异常值，如 `X[X > 0]` 只保留正特征。

- `np.where(condition, x, y)`：条件替换。  
  用处：处理缺失值或二值化特征，如 `np.where(X > 0, 1, 0)`。

## 10. 高级函数

- `np.clip(arr, a_min, a_max)`：裁剪值到范围。  
  用处：防止梯度爆炸，或标准化概率 [0, 1]。

- `np.unique(arr, return_counts=False)`：返回唯一值和计数。  
  用处：检查分类标签分布，或去重数据集。

## 11. 注意事项

- **向量化**：优先用 NumPy 函数避免循环，提高速度（e.g., `np.sum` 而非 for 循环）。
- **内存**：大数组注意内存使用，`dtype=float32` 节省空间。
- **兼容**：与 `sklearn`、`pandas` 无缝集成。

## 12. 机器学习示例

- **特征缩放**：
  ```python
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  X_scaled = (X - mu) / sigma
  ```
