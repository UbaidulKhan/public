import re

'''
Objective - find all call-outs: @UN, @UN_Women
'''

text10 = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women'

text11 = text10.split(' ')
# print(text11)


for w in text11:
  if(re.search('@[A-Za-z0-9_]+', w)):
  # if(re.search('@\w+', w)):
    print(w)
