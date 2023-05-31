import re

#
#------------------------------------------------------------------------------
#  Copyright(c):
#   Ubaidul Khan / ubaidul(dot)khan(at)gmail(dot)com
#   
#------------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------
#  References
#    https://pynative.com/python-regex-capturing-groups/   
#
#------------------------------------------------------------------------------
#
# This script demonstrates regular expression capture groups
#------------------------------------------------------------------------------



release_version_string = "20.16.3"
  
#
## This pattern works
reg_pattern = re.compile(r'(\d{2})\.(\d{2})\.?(\d{1,2})?')

version_string = []
version_string_string = ""

print(f' \-_-/ Input string: {release_version_string}')

result = reg_pattern.search(release_version_string)


'''
if(result.group(0)):
  print(f'  \-_-/ Group zero: {result.group(0)}')
'''

if(result.group(1)):
  # print(f'  \-_-/ Group one: {result.group(1)}')
  version_string.append(result.group(1))
    
if(result.group(2)):
  # print(f'  \-_-/ Group two is:: {result.group(2)}')
  version_string.append('.')
  version_string.append(result.group(2))
    
if(result.group(3)):
  # print(f'  \-_-/ Group three is:: {result.group(3)}')
  version_string.append(result.group(3))

for ele in version_string:
  version_string_string += ele

# version_float = float(version_string_string)

version_float = round(float(version_string_string), 4) 
print(f'  \-_-/ Version string constructed: {version_float}')

