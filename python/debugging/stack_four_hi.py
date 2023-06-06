import inspect


def make_function_name_table(name):
 

  to_return = None
  
  if(name): 

    #
    ## Lookup calling function name in stack
    curframe = inspect.currentframe()
    call_frame = inspect.getouterframes(curframe, 2)
 
    my_name = call_frame[0][3]
    print(f'  | \-(mfnt) My name is: {my_name}')
 
    my_parent = call_frame[1][3]
    print(f'  | \-(mfnt) My parent\'s name is: {my_parent}')

    my_great_grand_parent = call_frame[2][3]
    print(f'  | \-(mfnt) My parent\'s name is: {my_great_grand_parent}')

    
    # my_name, my_parents_name, my_grand_parents_name = stack_inspector()
  
    func_name_dict = {}

    #
    ## Take the name received and split it by underscore(_)
    name_split = name.split('_')
 
 
 
    acronym = ""
    word_pos = 0
    for word in name_split:
      #print(f'|  \_(ld) Word in {word_pos} the : {word} ')
      #print(f'|  \_(ld) First letter in word is {word[0]}')
      acronym = acronym + word[0]
      word_pos += 1
 
    #func_name_dict = {'make_function_name_table':'mfnt', }
    func_name_dict[name] = acronym
  
    print(f'  | \_(mfnt) Function name table: {func_name_dict}')

  else:
    print(f'  | \-(mfnt) Nothing to process')
    to_return = None


  return(to_return) 


#  
def stack_inspector():
  
  my_ancestor_count = 0
 
  my_parents_name = my_grand_parents_name = my_great_grand_parent = None
  
  curframe = inspect.currentframe()
  call_frame = inspect.getouterframes(curframe, 2)

  #
  ## Determine a name of the current function name & acronym
  my_name = curframe.f_code.co_name
  print(f'  |  \_(si) My name is: {my_name}')
  make_function_name_table(my_name)
 
  #
  ## Determine a name of the parent function name & acronym
  my_parents_name = call_frame[1][3]
  print(f'  | \_(si) Function that called me: {my_parents_name}')
  make_function_name_table(my_parents_name)

  #
  ## Determine a name of the grand parent function name & acronym
  my_grand_parents_name = call_frame[2][3]
  make_function_name_table(my_grand_parents_name)

  #
  ## Determine a name of the great grand parent function name & acronym
  try:
    my_great_grand_parent = call_frame[3][3]
    
  except Exception as e:
    print("  | \_(si) Range out of bounds - exiting")

  finally:
    return(my_parents_name, my_grand_parents_name, my_great_grand_parent)


def d():
  
  curframe = inspect.currentframe()
  call_frame = inspect.getouterframes(curframe, 2)
  print(f'\n call_frame type: {type(call_frame)}')
  
  call_frame_len = len(call_frame)
  print(f'\n >> Call Frame: \n\t {call_frame}')
  print(f'\n >> Call Frame length: \n\t {call_frame_len}')

  
  print("this is: ** d **")
  

def c():
  
  curframe = inspect.currentframe()
  call_frame = inspect.getouterframes(curframe, 2)
  print(f'\n call_frame type: {type(call_frame)}')
  
  call_frame_len = len(call_frame)
  print(f'\n >> Call Frame: \n\t {call_frame}')
  print(f'\n >> Call Frame length: \n\t {call_frame_len}')



  print("this is: ** c ** calling d")
  d()
  

def b():

  curframe = inspect.currentframe()
  call_frame = inspect.getouterframes(curframe, 2)
  print(f'\n call_frame type: {type(call_frame)}')
  
  call_frame_len = len(call_frame)
  print(f'\n >> Call Frame: \n\t {call_frame}')
  print(f'\n >> Call Frame length: \n\t {call_frame_len}')


  print("this is: ** b ** calling c")
  c()
  
  
def a():
  
  curframe = inspect.currentframe()
  call_frame = inspect.getouterframes(curframe, 2)
  print(f'\n call_frame type: {type(call_frame)}')
  
  call_frame_len = len(call_frame)
  print(f'\n >> Call Frame: \n\t {call_frame}')
  print(f'\n >> Call Frame length: \n\t {call_frame_len}')


  print("this is: ** a ** calling b")
  b()
  
  
def main():
  
  # stack_inspector()
  
  curframe = inspect.currentframe()
  call_frame = inspect.getouterframes(curframe, 2)
  print(f'\n call_frame type: {type(call_frame)}')
  
  call_frame_len = len(call_frame)
  print(f'\n >> Call Frame: \n\t {call_frame}')
  print(f'\n >> Call Frame length: \n\t {call_frame_len}')

  
  print("this is: ** main ** calling a")
  a()
  

if __name__ == "__main__":
  main()  