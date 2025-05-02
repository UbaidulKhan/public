
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words
from collections import Counter

#
## 2022-09-14, Ubaidul Khan
#
##Intended purpose:
#  1) For a given misspelled word and n-gram size:
#    traverse nltk provided words, for words that
#    start with the same letter as the supplied
#    incorrect word
#
#  2) Calculate the jaccard distance(jc), return a list 
#    of words with highest JC value
#
###
#  
#
## References:
#
#   Jaccard Distance:
#     
#     https://mariogarcia.github.io/blog/2021/04/nlp_text_similarity.html  [ Following ]
#     https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/   [ Following ]
#     https://predictivehacks.com/spelling-recommender-with-nltk/
#     https://www.geeksforgeeks.org/spelling-correction-using-k-gram-overlap/
#     
###
#

def get_ngrams(word, n):
    return set(ngrams(word,n=n))

def get_recommendation_for(bad_word, n=3, threshold=0.39):

    # Create an empty dictionary
    correct_words_list = {}
    
    # we don't want stop-words to be processed
    if len(bad_word) < 3:
        return bad_word
        
    score = 0
    spellings = words.words()
        
    for word in spellings:
      if(word.startswith(bad_word[0])):
        # print(f'Comparing: {word} to {bad_word}')
        jacc_distance = (1 - jaccard_distance(set(bad_word), set(word)))
        if(jacc_distance > threshold):
          # print(f'Jaccard distance of {word} is: {jacc_distance}')
          correct_words_list[word] = jacc_distance
          # print(correct_words_list)
        else:
          continue

    # print(f'Unsorted dictionary is: {correct_words_list}')
    sorted_correct_words_list = sorted(correct_words_list.items(), key=lambda dict: dict[1], reverse=True)
    # print(f'Sorted dictionary is: {sorted_correct_words_list}')
    return(sorted_correct_words_list[0:5])

    
gram_size=3
threshold=0.625
word = "Hindut"
most_likely_correct = get_recommendation_for(word, gram_size, threshold)
print(f'\n{word} misspelled, possibly corrections: {most_likely_correct}')


gram_size=3
threshold=0.625
word = "Bangaldesh"
most_likely_correct = get_recommendation_for(word, gram_size, threshold)
print(f'\n{word} misspelled, possibly corrections: {most_likely_correct}')