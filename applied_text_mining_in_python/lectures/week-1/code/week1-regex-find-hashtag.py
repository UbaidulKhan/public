'''
Objective is find all hash-tags, call-outs
'''
tweet = "@nltk Text analysis is awesome! #regex #pandas #python"
text = tweet.split()

for i in text:
  if(i.startswith('#')):
    print("Here is a  hash-tag")
    print(i)
  if(i.startswith('@')):
    print("Here is a call-out")
    print(i)
