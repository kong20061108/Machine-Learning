from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#可视化过程

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

#print(diabetes.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#创建模型
model = LinearRegression()
model.fit(X_train, y_train)  #训练模型

y_pred = model.predict(X_test)  #预测
print(y_pred)
print(model.coef_)  #打印回归系数
print(model.intercept_)  #打印截距
print("均方误差: %.2f" % mean_squared_error(y_test, y_pred))
print('R^2: %.2f' % r2_score(y_test, y_pred))
#可视化
plt.scatter(y_test, y_pred) #真实值与预测值对比
plt.xlabel("True Values") #X轴标签
plt.ylabel("Predictions")
plt.axis('equal') 
plt.axis('square')
plt.xlim(plt.xlim())  
plt.ylim(plt.ylim())
_ = plt.plot([-100, 500], [-100, 500])
plt.show()