import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from pprint import pprint
#
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#
## References:
#
#  https://www.nltk.org/howto/wordnet.html
#

pprint(wn.synset('bound.a.01').tree(lambda s:s.also_sees()))

print(f'\n-----------------------------------------------')

pprint(wn.synset('dependent.a.01').tree(lambda s:s.also_sees()))
print(f'\n-----------------------------------------------')

pprint(wn.synset('child.a.01').tree(lambda s:s.also_sees()))


# pprint(wn.synset('restrained.a.01').tree(lambda s:s.also_sees()))
