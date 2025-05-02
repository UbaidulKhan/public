import spacy
nlp = spacy.load("en_core_web_sm")

with open ("data/wiki_us.txt", "r") as f:
  text = f.read()

# print(text)

#**************************************
# 2.2. Creating a Doc Container
#**************************************
doc = nlp(text)
words = text.split()[:10]

'''
print(doc)
print(len(text))
print(len(doc))

for token in text[0:10]:
  print(token)

for token in doc[0:10]:
  print(token)

for token in text.split()[:10]:
  print(token)

i=5
for token in doc[i:8]:
  print (f"SpaCy Token {i}:\n{token}\nWord Split {i}:\n{words[i]}\n\n")
  i=i+1
'''


#**************************************
# 2.3 Sentence Boundary Detection(SBD)
#**************************************
'''
for sent in doc.sents:
  print(sent)
'''

sentence1 = list(doc.sents)[0]
print(sentence1)


#
## Accessing token attributes/meta-data
##
## GPE - Geo Political Entity - 384
## _iob_ : I indicates the entity is inside a larger entity
##
token2 = sentence1[2]
print(token2)
print(token2.text)
print(token2.left_edge)
print(token2.right_edge)
print(token2.ent_type)
print(token2.ent_type_)
print(token2.ent_iob_)


#
## Lemmatization
##
print(token2.lemma_)

print(f"sentence 12 original form: {sentence1[12]} ")
print(f"sentence 12 lemma: {sentence1[12].lemma_} ")
