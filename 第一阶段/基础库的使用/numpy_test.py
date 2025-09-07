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