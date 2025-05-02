import re

shop="hello seattle what have you got"
regex = r'\w+'
list1=re.sub(regex, '', shop)
print(list1)
