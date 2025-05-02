import numpy as np
import pprint
from pprint import PrettyPrinter


#
## References:
#
#  https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-a-numpy-array-and-specify-the-index-column-and-column-headers/
#
#



def two_d_array_concatenation():
  print('\n+----------------------------------------------')
  print('|  Executing two_d_array_concatenation()')

  
  # pp = pprint.PrettyPrinter(indent=10)
  
  arr = np.array([[1, 2], [3, 4]])
  
  # print(f'   array is of type: {type(arr)}\n')
  # pp.pprint(arr)
  # print(f'   Array consits of: \n\t')
  # pp.pprint(arr)
  
  print('\n\t' + str(arr).replace('\n', '\n\t'))
  
  
  
  print(f'\n   array has the shape: {arr.shape}')
  new_col = np.array([[5, 6]])
  new_arr = np.concatenate((arr, new_col), axis=0)
  
  print("   After row-wise addition:\n")
  # pp.pprint(new_arr)
  print('\t' + str(new_arr).replace('\n', '\n\t'))



def three_d_array_concatenation():
  pp = pprint.PrettyPrinter(indent=10)

  print('\n+----------------------------------------------')
  print('|  Executing three_d_array_concatenation()')
  
  x = np.array([[3, 4], [5, 6]])
  print('\n   x array is:\n')
  print('\t' + str(x).replace('\n', '\n\t'))

  # pp.pprint(x)
  
  y = np.array([[7, 8]])
  print('\n   array is y:\n')
  print('\t' + str(y).replace('\n', '\n\t'))
  
  #
  ## Insert a name for the column
  

  new_arr = np.concatenate((x, y.T), axis=1)
  print('\n   Concatenated(x and y) array is:\n')
  print("   Concatenated after column-wise addition:\n")
  
  col_names = ['a', 'b', 'c']
  array = np.vstack((col_names,new_arr))
  

  # pp.pprint(new_arr)
  print('\t' + str(new_arr).replace('\n', '\n\t'))
  
  print('\n---------------------------')
  print('\t' + str(array).replace('\n', '\n\t'))
  print('\n')



print("\n Calling two_d_array_concatenation")
two_d_array_concatenation()

print("\n Calling three_d_array_concatenation")
three_d_array_concatenation()