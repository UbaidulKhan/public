import nltk
from nltk.util import ngrams

# text1 = 'Hello my name is Jason'
text1 = 'HelloMyNameIsJason'
text2 = 'My name is not Mike'

n = 3
trigrams1 = ngrams(text1.split(), n)
print(f'trigrams1 type is: {type(trigrams1)}')
trigrams2 = ngrams(text2.split(), n)

'''
print(f'Trigram-1 is: {trigrams1}')
print(f'Trigram-2 is: {trigrams2}')
'''

g_count = 0
for grams in trigrams1:
  print(f'Trigram {g_count} is: {grams}')
  g_count += 1

