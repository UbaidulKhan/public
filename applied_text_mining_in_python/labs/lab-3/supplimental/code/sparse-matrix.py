#
#

import numpy as np
from scipy.sparse import csr_matrix

#
## References:
#  w3schools - https://www.w3schools.com/python/scipy/scipy_sparse_data.php
#  udacity - https://www.youtube.com/watch?v=Lhef_jxzqCg                                      ****
#
## Objective: 
##  1) Demonstrate the use of Compressed Sparse Matrix(CSR)
##  2) Explain the compressions
##  3) 
##

#------ This code is working -------#
'''
array1 = np.array(['a', '0', 'b'])
array2 = np.array(['c', 'd', 'e'])
array3 = np.array([0, 0, 'f'])
arrayOfArrays = np.array([array1, array2, array3])

print(arrayOfArrays)
print(csr_matrix(arrayOfArrays))

 
# arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
# arr = np.array([['a', 'x', 'b'], ['c', 'd', 'e'], ['y', 'k', 'f']])
# print(csr_matrix(arr))


#<---------- this does not work ------------->
# np_array = np.zeros((0), dtype=mtype)
# np_array = np.append(np_array, np.array([('first', '10', '11')], dtype=mtype))
# np_array = np.append(np_array, np.array([('second', '20', '21')], dtype=mtype))


mtype = 'S10, i4, i4'

np_array = np.zeros((0), dtype=mtype)
np_array = np.append(np_array, np.array([('first', 10, 11)], dtype=mtype))
np_array = np.append(np_array, np.array([('second', 20, 21)], dtype=mtype))
print(np_array)


#
## Convert the np array to CSR
# this_csr = csr_matrix(np_array)

'''

#<---------- this code is working ------------->

mtype = 'i4, i4, i4'
np_array = np.zeros((0), dtype=mtype)

array1 = np.array([1, 2, 3])
array2 = np.array([0, 4, 0])
array3 = np.array([0, 5, 6])
arrayOfArrays = np.array([array1, array2, array3])

print(f'\nnumPy array:\n{arrayOfArrays}')

arrayOfArrays_csr = csr_matrix(arrayOfArrays)
print(f'\nCSR Representation:\n{arrayOfArrays_csr}')


#
## Produces the following output:
#
#  CSR - Compressor Sparsed Row is a respresenation that removes all zeros
#
#   numPy array:
# 
#    Column number
#      0 | 1 |  2    
#-------------------------
#  0  [1 | 2 | 3]   | x  |
#  1  [0 | 4 | 0]   | y  |
#  2  [0 | 5 | 6]   | z  |
#              
#   
#         CSR Representation:
#-----------------------------------------
#       (0, 0)	1   (row 0, column 0)
#       (0, 1)	2   (row 0, column 1)
#       (0, 2)	3   (row 0, column 2)
#       (1, 1)	4   (row 1, column 1)
#       (2, 1)	5   (row 2, column 1)
#       (2, 2)	6   (row 2, column 2)
#
#  The CSR represenation requires three vectors:
#
#   Index:             0  1  2  3  4  5   
#
#   Value(non-zero): [ 1  2  3  4  5  6 ]
#
#   Column:          [ 0  1  2  1  1  2 ]
#
#   RowPtr:          [ 0, 4, 5 ]
#   
