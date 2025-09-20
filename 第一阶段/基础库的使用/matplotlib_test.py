#matplotlib库在机器学习中的应用
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-1,1,50)  #linspace表示在-1到1之间取50个点
y1=2*x+1
y2=x**2
plt.figure() #figure函数用于创建一个图形实例
plt.plot(x,y1)  #plot函数用于绘制线条
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--') #linestyle表示线条样式
plt.show()
plt.figure(num=3,figsize=(8,5)) #num表示图形编号，figsize表示图形大小
plt.plot(x,y1) 
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--') #linestyle表示线条样式
plt.xlim((-1,1)) #x轴范围
plt.ylim((-0.2,2.2)) #y轴范围
plt.xlabel('I am x') #x轴标签
plt.ylabel('I am y') #y轴标签
new_ticks=np.linspace(-1,1,5) #在-1到1之间取5个点
print(new_ticks)
plt.xticks(new_ticks) #设置x轴刻度
plt.yticks([-2,0,1.22,2.0],['really bad','bad','normal','good']) #设置y轴刻度
#plt.yticks([-2,0,1.22,2.0],['really bad','bad','normal','good'],rotation=45) #设置y轴刻度并旋转45度
plt.grid() #显示网格
#plt.scatter(x,y1) #散点图
plt.scatter(x,y2) #散点图
plt.show()
plt.figure()
plt.plot(x,y1,label='linear line')
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--',label='quadratic line') #linestyle表示线条样式
plt.xlim((-1,1)) #x轴范围
plt.ylim((-0.2,2.2)) #y轴范围
plt.xlabel('I am x') #x轴标签
plt.ylabel('I am y') #y轴标签
new_ticks=np.linspace(-1,1,5) #在-1到1之间取
print(new_ticks)
plt.xticks(new_ticks) #设置x轴刻度      
plt.yticks([-2,0,1.22,2.0],['really bad','bad','normal','good']) #设置y轴刻度
plt.grid() #显示网格
plt.legend(loc='best') #显示图例，loc表示图例位置
plt.show()