# import cloud_init_gen
import cloud_init_gen

# List all the attributes and methods of the cloud_init_gen module
# classes = dir(cloud_init_gen)

# Get help/documentation for a specific class or method
# help(cloud_init_gen.some_class)
# help(cloud_init_gen.some_method)

all_attributes = dir(cloud_init_gen)

# Filter the attributes to get only the class names
class_names = [item for item in all_attributes if isinstance(getattr(cloud_init_gen, item), type)]

print(" This package contains the following class(s):\n\n")


# Print the class names
for each_class in class_names:
  print(f'\n -- Class name: {each_class} -- ')
  print(f'   -- Methods supported by {each_class} --')
  for each_method in dir(each_class):
    
    # print(f'    Checking method: {each_method} ')
    
    # print(f'Class.method docs:')
    # help(each_class.each_method)
    
    if("__" in each_method):
      # print(f'      Private method found: {each_method}\n')
      continue
    
    else:
      # print(f'      Showing help for: {each_method}\n')
      class_method_str = str(each_class) + '.' + str(each_method)
      # help(each_class.each_method)
      print(f'\n      Class_Method: {class_method_str}\n')
      # help(class_method_str)
