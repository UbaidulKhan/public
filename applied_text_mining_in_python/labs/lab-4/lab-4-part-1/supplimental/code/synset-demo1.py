import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
import pprint

#
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#
## References:
#
#  https://opensource.com/article/20/8/nlp-python-nltk
#
#  https://www.nltk.org/howto/wordnet.html
#
#  https://alvinntnu.github.io/python-notes/corpus/wordnet.html
#

def synset_info(synset):
  print("\n")
  print("+------------------------------------------------------------")
  print("|  \033[1m --- synset_info --- \033[0m")
  print("+------------------------------------------------------------")
  print(f'  Synset operating on: {synset}')
  
  print("  Name", synset.name())
  print("  POS:", synset.pos())
  print("  Definition:", synset.definition())
  print("  Examples:", synset.examples())
  print("  Lemmas:", synset.lemmas())
  print("  Antonyms:", [lemma.antonyms() for lemma in synset.lemmas() if len(lemma.antonyms()) > 0])
  print("  Hypernyms:", synset.hypernyms())
  print("  Instance Hypernyms:", synset.instance_hypernyms())
  print("  Part Holonyms:", synset.part_holonyms())
  print("  Part Meronyms:", synset.part_meronyms())

# synsets = wn.synsets('code')
synsets = wn.synsets('house')

print(f'synsets is of type: {type(synsets)}')

for synset in synsets:
  synset_info(synset)


def hypernyms(synset):
    return synset.hypernyms()

synsets = wn.synsets('soccer')
for synset in synsets:
    print(synset.name() + " tree:")
    # pprint(synset.tree(rel=hypernyms))
    pprint(wn.synset('bound.a.01').tree(lambda s:s.also_sees()))
    print()