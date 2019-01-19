import numpy as np
from scipy.sparse import coo_matrix


# a = np.matrix([[1, 2], [3, 4]])
# b = np.matrix([[5, 6, 7], [8, 9, 10]])
# c = np.concatenate((a, b), axis=1)
# np.random.shuffle(c)
#
# print(c)
# a = c[:, :2]
# print(c[:, 2:-1])
# print(a)
# print(type(a))

# print(c.shape)
# #
# x = np.matrix([[1, 2], [4, 3], [5, 6]], dtype=np.float)
# print(x.shape)
# x /= x.sum(1)
# print(x)
# print(x.shape[1])
# y = np.matrix([[1, 2], [4, 3], [5, 6]], dtype=np.float)
# y /= y.sum(1)
# y =  y.reshape(-1, 1)
# print(y)

#
# z = np.array([[1, 2, 3, 4],
#         [5, 6, 7, 8],
#         [9, 10, 11, 12],
#         [13, 13, 13],
#         [11, 11, 11]])
# print(z)
# id = range(2)
# id2 = range(2,3)
# id3 = range(3,5)
# list_id = list(id2) + list(id3) + list(id)
# print(list_id)
# z = z[list_id]
# print(z)

x = np.matrix([[1, 2], [4, 3], [5, 6]], dtype=np.float)
index = np.arange(len(x)).reshape(-1, 1)
x = np.append(x, index, axis=1)
new_index = x[:, -1]
print(new_index)
print(type(new_index))



numberList = [1, 2, 3]
strList = ['one', 'two', 'three']

numberList2 = [4, 5, 6]
strList2 = ['ne', 'wo', 'ree']



# Two iterables are passed
result, result2 = zip([numberList, numberList2], [strList, strList2])

# Converting itertor to set
print(list(result))
print(list(result2))