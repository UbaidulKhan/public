#------------------------------------------------------------------------------
#  Copyright(c):
#   Ubaidul Khan / ubaidul(dot)khan(at)gmail(dot)com
#   
#------------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------
#  References
#   https://www.stefaanlippens.net/python_inspect/
#
#------------------------------------------------------------------------------
#
# This script demonstrates python stack access - fetching function name  and
# name of the calling function name.
#------------------------------------------------------------------------------
import inspect



#
## header_printer(func_name, func_abb)
#------------------------------------------------------------------------------
# 
#
#------------------------------------------------------------------------------
#
# hp - function acronym
# 
## 
def debugging_header_printer(func_name, func_abb):
  
  if(global_debug_on):
    print(f'\n  +--------------------------------------------------------------------+')
    print(f'  | <<<<<<<<<<<<< {func_name} {func_abb} -------------          ')
    print(f'  +--------------------------------------------------------------------+')

#
## trailer_printer(func_name, func_abb)
#------------------------------------------------------------------------------
# 
#
#------------------------------------------------------------------------------
#
# hp - function acronym
# 
## 
def debugging_trailer_printer(func_name, func_abb):
  
  if(global_debug_on):
    print(f'  +--------------------------------------------------------------------+')
    print(f'  | ------------- {func_name} {func_abb} >>>>>>>>>>>>>         ')
    print(f'  +--------------------------------------------------------------------+\n')



#
## debugging_printer(func_abb, message, release_url)
#------------------------------------------------------------------------------
# 
#
#------------------------------------------------------------------------------
#
# dp - function acronym
# 
## 
def debugging_printer(func_abb, message, release_url):
  ''' 
  print(f'\n  +--------------------------------------------------------------------+')
  print(f'  | <<<<<<<<<<<<< debugging_printer(dp) -------------          ')
  print(f'  +--------------------------------------------------------------------+')
  ''' 
  
  if(global_debug_on):
    print(f'  | \-{func_abb} {message}: {release_url}')

  ''' 
  print(f'  +--------------------------------------------------------------------+')
  print(f'  | ------------- debugging_printer(dp) >>>>>>>>>>>>>')
  print(f'  +--------------------------------------------------------------------+\n\n')
  ''' 




#
## ANCILIRY: make_function_name_table()
##-----------------------------------------------------------------------------
#
# Provide function takes a function-name and generates an acronym for the
# function. Then this acronym is added to a dictionary.
#
## References:
##
##  https://www.geeksforgeeks.org/add-a-keyvalue-pair-to-dictionary-in-python/
##
## To do:
##  1) Construct table
##  2) Write a look up function
#
# mfnt - method acronym
#
def make_function_name_table(name):
 
 

  print(f'\n  +--------------------------------------------------------------------+')
  print(f'  | <<<<<<<<<<<<< \033[1m - make_function_name_table(mfnt) - \033[0m -------------')
  print(f'  +--------------------------------------------------------------------+') 
  
  to_return = None
  
  if(name): 
    
 
    #
    ## Lookup calling function name in stack
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
 
    my_name = calframe[0][3]
    print(f'  | \-(mfnt) My name is: {my_name}')
 
    my_parent = calframe[1][3]
    print(f'  | \-(mfnt) My parent\'s name is: {my_parent}')

    my_great_grand_parent = calframe[2][3]
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

  print(f'  +--------------------------------------------------------------------+')
  print(f'  | ------------- make_function_name_table(mfnt) >>>>>>>>>>>>>')
  print(f'  +--------------------------------------------------------------------+')

  return(to_return) 

#
## ANCILIRY: stack_inspector()
##-----------------------------------------------------------------------------
#
#  
def stack_inspector():
 
  my_parents_name = my_grand_parents_name = my_great_grand_parent = None
  
  curframe = inspect.currentframe()
  calframe = inspect.getouterframes(curframe, 2)

  print("\n")
  print(f'\n  +--------------------------------------------------------------------+')
  print(f'  | <<<<<<<<<<<<< \033[1m - stack_inspector()(si) - \033[0m -------------')
  print(f'  +--------------------------------------------------------------------+') 
  

  #
  ## Determine a name of the current function name & acronym
  my_name = curframe.f_code.co_name
  print(f'  |  \_(si) My name is: {my_name}')
  make_function_name_table(my_name)
 
  #
  ## Determine a name of the parent function name & acronym
  my_parents_name = calframe[1][3]
  print(f'  | \_(si) Function that called me: {my_parents_name}')
  make_function_name_table(my_parents_name)

  #
  ## Determine a name of the grand parent function name & acronym
  my_grand_parents_name = calframe[2][3]
  make_function_name_table(my_grand_parents_name)

  #
  ## Determine a name of the great grand parent function name & acronym
  try:
    my_great_grand_parent = calframe[3][3]
    
  except Exception as e:
    print("  | \_(si) Range out of bounds - exiting")

  finally:
    return(my_parents_name, my_grand_parents_name, my_great_grand_parent)

  # return(my_parents_name, my_grand_parents_name)

  print(f'  +--------------------------------------------------------------------+')
  print(f'  | ------------- stack_inspector()(si) >>>>>>>>>>>>>')
  print(f'  +--------------------------------------------------------------------+')

  


#
## main()
##-----------------------------------------------------------------------------
#
#
def main():
  
  # print("\033[1m ")
  print("\n")
  print("+------------------------------------------------------------")
  print(f'|  \033[1m - main() - \033[0m')
  print("+------------------------------------------------------------")


  stack_inspector()


  
if __name__ == "__main__":
  main()  
