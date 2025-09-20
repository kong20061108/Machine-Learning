import numpy as np

x=np.array([[2,3],[1,4],[99,2]])
print(x)
print(f'数组形状为：{x.shape}')
print(f'数组的维度为{x.ndim}')
print(f'数组的元素总数为{x.size}')
#[[[1]]]这种是三维
print(x[0][1])
print('\n')

ary1=np.zeros((3,3))
print(ary1)
print('\n')
ary2=np.full((2,2),9)
print(ary2)
print('\n')
ary3=np.arange(2,10,2)
print(ary3)
print('\n')

ary4=np.linspace(0,1,11)
print(ary4)
print('\n')

a=np.array([1,2,3])
b=np.array([0,1,2])
print(np.dot(a,b))

a=np.array([[1,2,3],
            [1,1,1]])
b=np.array([[0,1],
            [0,0],
            [1,1]])
print(np.dot(a,b))
print('\n')
t=np.array([[1,2,3],
            [4,5,6]])
print(np.transpose(t))
print(f't数组的总体均值为：{np.mean(t)}')
print(f't数组的列均值为：{np.mean(t,axis=0)}')
print(f't数组的行体均值为：{np.mean(t,axis=1)}')
print(f'数组总和为：{t.sum()}')
print(f'数组列和为：{t.sum(axis=0)}')
print(f'数组行和为：{t.sum(axis=1)}')

#切片

a=np.array([[1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]])
#逗号前面表示行，后面表示列
print(a[0:2,1:3])
print(a[:,1:3])
print(a[1,:])
print(a[1:3,:])
print(a[[0,1,2],[0,1,2]])#取对角线元素

print(a>5)
print(a[a>5])#取出大于5的元素
print('\n')
#花式索引
b=np.array([[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]])    
print(b[[0,1,2,3],[0,2,0,1]])#取出1，6，7，11
print(b[[0,0,3,3],[0,2,0,1]])
print('\n')
#数组的形状修改
c=np.array([[1,2,3,4,5,6],
            [7,8,9,10,11,12]])
print(c.reshape(3,4))
print(c.reshape(6,2))
print(c.reshape(12,1))
print(c.reshape(1,12))
print('\n')
#数组的合并
a=np.array([[1,2,3],    
            [4,5,6]])
b=np.array([[7,8,9],
            [10,11,12]])
print(np.vstack((a,b)))#垂直合并
print(np.hstack((a,b)))#水平合并
print('\n')
#数组的分割
a=np.array([[1,2,3,4,5,6],
            [7,8,9,10,11,12]])
print(np.hsplit(a,3))#水平分割
print(np.vsplit(a,2))#垂直分割
print('\n')
#数组的复制
a=np.array([[1,2,3],
            [4,5,6]])
b=a#浅复制
c=a.copy()#深复制
a[0,0]=100
print(a)
print(b)
print(c)
print('\n')
#数组的遍历
a=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
for row in a:
    print(row)
print('\n')
for column in a.T:
    print(column)
print('\n')
for item in a.flat:
    print(item)
print('\n')
#数组的排序 
a=np.array([[4,3,5],
            [1,6,2]])
print(np.sort(a))#按行排序
print(np.sort(a,axis=0))#按列排序
print(np.argsort(a))#返回排序后的索引
print(np.argsort(a,axis=0))#按列返回排序后的索引