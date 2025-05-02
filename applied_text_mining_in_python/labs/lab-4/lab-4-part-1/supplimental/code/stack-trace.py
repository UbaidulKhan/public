
import inspect

def child():
  
   curframe = inspect.currentframe()
   
   print(f'Frame info: {curframe}')
   
   calframe = inspect.getouterframes(curframe, 2)

   my_name = curframe.f_code.co_name
   my_parent = calframe[1][3]
   my_grand_parent = calframe[2][3]

   print('\n')
   print(f'  >> my name: {my_name}')
   print(f'  >> my parents name: {my_parent}')
   print(f'  >> my grand parents name: {my_grand_parent}')

   print('\n')
  

def parent():
  child()


def main():
  parent()
  
#
if __name__ == "__main__":
  main()