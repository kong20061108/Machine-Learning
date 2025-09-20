import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 常量定义
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 200

# 2. 数据加载与预处理函数
def load_and_preprocess_data():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    
    df = pd.DataFrame(X, columns=dataset.feature_names)
    df['target'] = y
    
    print("特征名称:", dataset.feature_names)
    print("目标名称:", dataset.target_names)
    print("数据前五行:\n", df.head())
    print("数据描述:\n", df.describe())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df, dataset

# 3. 模型训练与评估函数
def train_and_evaluate_model(X, y, dataset):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"模型训练失败: {e}")
        return None
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.2f}")
    print("分类报告:\n", classification_report(y_test, y_pred, target_names=dataset.target_names))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5 折交叉验证得分: {cv_scores}")
    print(f"平均交叉验证得分: {cv_scores.mean():.2f} (± {cv_scores.std() * 2:.2f})")
    
    return model, y_test, y_pred, X_train, X_test

# 4. 可视化函数
def plot_confusion_matrix(y_test, y_pred, target_names):
    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵形状: {cm.shape}")  # 调试：检查矩阵维度
    n_classes = cm.shape[0] if cm.ndim > 1 else len(np.unique(y_test))  # 动态确定类别数
    
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # 添加数值标签
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0),
                     ha='center', va='center', color='red')
    
    plt.xticks(range(n_classes), target_names[:n_classes], rotation=45)
    plt.yticks(range(n_classes), target_names[:n_classes])
    plt.show()

def plot_feature_scatter(df, feature1='sepal length (cm)', feature2='sepal width (cm)'):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df[feature1], df[feature2], c=df['target'], cmap='viridis')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"{feature1} vs {feature2} by Target")
    plt.colorbar(scatter, label='Target Class')
    plt.show()

# 5. 主程序
if __name__ == "__main__":
    X, y, df, dataset = load_and_preprocess_data()
    model, y_test, y_pred, X_train, X_test = train_and_evaluate_model(X, y, dataset)
    if model is None:
        exit()
    
    plot_confusion_matrix(y_test, y_pred, dataset.target_names)
    plot_feature_scatter(df)
    
    joblib.dump(model, 'logistic_regression_model.joblib')
    print("模型已保存为 'logistic_regression_model.joblib'")