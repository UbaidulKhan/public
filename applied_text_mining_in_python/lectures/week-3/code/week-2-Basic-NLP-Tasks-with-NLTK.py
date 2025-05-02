#
## Coursera - Applied Text Mining in Python
##   Week 2
##   Basic Natural Language Processing
##   Basic NLP Tasks with NLTK

#
import nltk
from nltk.book import *
from nltk.corpus import state_union
from nltk import sent_tokenize


print("-----------------------------")
print(f"sent7 contains:\n\t {sent7}")

print("-----------------------------")
print(f"text7 contains:\n\t {text7}")

print("-----------------------------")
text7_1 = set(text7)
print(f"text7 type: {type(text7_1)}")
print(f"text7 length: {len(text7_1)}")
'''

#
## Text 7_1 is a large set contaning 100676 tokens

for e in text7_1:
    print(e)
    print("\n----------------------\n")
'''



#
## Frequency distribution
print("\n--------------- Frequency distribution ---------------")
dist = FreqDist(text7)
print(f"text7 frequency of words: {len(dist)}")
print("-----------------------------")

#
## Frequency of words
print("\n--------------- Frequency of words ---------------")
vocab1 = dist.keys()
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
list(vocab1)[:10]

print(dist['four'])

freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
print(freqwords)

#
## Stemming
print("\n--------------- Stemming ---------------")
porter = nltk.PorterStemmer()
input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
print(f"\nPorter stemmer stemming{input1}\n")
for t in words1:
  print(f"stemming {t} is: {porter.stem(t)}")


#
## Lemmetization
print("\n--------------- Lemmetization ---------------")
udhr = nltk.corpus.udhr.words('English-Latin1')
WNlemma = nltk.WordNetLemmatizer()

print(f"\nUDHR first 20 words\n\t: {udhr[:20]}")
for t in udhr[:20]:
  print(f"Lemmetizing {t} is: {WNlemma.lemmatize(t)}")


#
## Tokenization
print("\n--------------- Tokenization ---------------")
text11 = "Children shouldn't drink a sugary drink before bed."
text11_split = text11.split(' ')
text11_split_len = len(text11_split)

print(f"Python split text11: {text11_split}")
text11_nltk_tokenized = nltk.sent_tokenize(text11)
text11_nltk_tokenized_len = len(text11_nltk_tokenized)
print(f"NLTK split text11: {text11_nltk_tokenized}")


#
## Finding Sentence Boundaries
some_text = "Today is Wednesday, July 6th. It's hot outside and I would rather be swimming"
print(f"\nSentence boundary detection for:\n\t {some_text}")
sentences = sent_tokenize(some_text)
print(f"\nSentence tokenized to: {sentences[0]}")
print(f"\nSentence tokenized to: {sentences[1]}")


#
## Advanced NLP Tasks with NLTK
#
## Parts of Speech(POS) tagging. This code detectcs all modals from
## the speech. Modal in grammar (of a verb form or auxiliary verb) 
## expressing a distinction of mood, such as that between possibility 
## and actuality. The modal auxiliaries in English include can, could, 
## may, must, need, ought, shall, should, will, and would
#
print("\n--------------- Parts of Speech(POS) tagging ---------------")
nltk.help.upenn_tagset('MD')
text13 = nltk.word_tokenize(text11)
print(f"\nPOS: \n\t {nltk.pos_tag(text13)}")

text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
print(f"\ntext4 is: {text14}")
print(f"\ntext14 POS: \n\t {nltk.pos_tag(text14)}")

text15 = nltk.word_tokenize("বিনামূল্যের Cramer says $Ford is a buy at this price")
print(f"\ntext15 is: {text15}")
print(f"\ntext15 POS: \n\t {nltk.pos_tag(text15)}")

#
## Parsing sentence structure
#
text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('mygrammar.cfg')
print(grammar1)

#
## Stemming
input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
print(words1)
porter = nltk.PorterStemmer()
for t in words1:
    print(f"{t} stemps from: {porter.stem(t)}")

