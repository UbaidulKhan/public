import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams


#
## Source:
#  https://mariogarcia.github.io/blog/2021/04/nlp_text_similarity.html
#

''' 
#
## similarty between these two sentences is: 0.0
sentence_1 = "aloha amigo is fun"
sentence_2 = "I am a go fan"
'''

''' 
#
## similarty between these two sentences is: 0.5
sentence_1 = "Today is Monday"
sentence_2 = "Today is Tuesday"
'''

''' 
#
## similarty between these two sentences is: 
''' 
sentence_1 = "I am a big fan"
sentence_2 = "I am a tennis fan"



def get_similarity_by_tokens(left, right, n):
    tokens_1 = set(nltk.word_tokenize(left)) # sequence of tokens
    print(f'Tokens in sentence 1 is: {tokens_1}')
    tokens_2 = set(nltk.word_tokenize(right)) # sequence of tokens
    print(f'Tokens in sentence 2 is: {tokens_2}')

    seq_1_grams = set(ngrams(nltk.word_tokenize(left), n=n))
    
    for gram1 in seq_1_grams:
      print(f'Tri-gram of *{seq_1_grams}*: {gram1}')
      
    seq_2_grams = set(ngrams(nltk.word_tokenize(right), n=n))
    
    print("\n----------------------------------------")
    
    for gram2 in seq_2_grams:
      print(f'Tri-gram of *{seq_2_grams}*: {gram2}')

    similarity = (1 - jaccard_distance(seq_1_grams, seq_2_grams))
    return (similarity)

gram_size = 2
similarity_by_tokens = get_similarity_by_tokens(sentence_1, sentence_2, gram_size)
print(f'\nSimilarity between {sentence_1} and {sentence_2} is: {similarity_by_tokens}')