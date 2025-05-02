import nltk
import pandas as pd
from pprint import pprint
import numpy as np
from nltk.stem import WordNetLemmatizer

#
## Sources found on the web:
#
''' 
   https://necromuralist.github.io/data_science/posts/assignment-2-introduction-to-nltk/

   https://github.com/raj-aditya007/Applied-Text-Mining-python-Coursera/blob/master/Assignment%2B2.py

   https://www.kaggle.com/code/askolkova/coursera-nlp-learning-assignment2/notebook
'''
##
#
# References:
#  https://www.nltk.org/howto/concordance.html
#


#
## Returns number of tokens (words and punctuation symbols) are in text_to_nltk_text
#
def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text_to_nltk_text)


#
## Returns number of tokens (words and punctuation symbols) are in text_to_nltk_text
#
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text_to_nltk_text))


#
## Returns number of unique tokens, after lemmatizing the verbs
#
def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text_to_nltk_text]

    return len(set(lemmatized))


#-------------------------- Question 1 --------------------------


#
## Returns number of unique tokens, after lemmatizing the verbs
#
def answer_one(unique_tokens, total_tokens):    
    
    return(unique_tokens/total_tokens)


#-------------------------- Question 2 --------------------------


#
## Return the lexical diversity of the given text input
## Resource: https://www.nltk.org/howto/concordance.html
#
def answer_two(text, token, moby_tokens):    
    whale_count = 0
    
    # return(len(text.concordance_list(token)))
    
    for iter_token in moby_tokens: 
        if((iter_token == 'whale') or (iter_token == 'Whale')):
            whale_count += 1
    
    return(whale_count)


#-------------------------- Question 3 --------------------------


#
## Return a tuple(word, occurance) of the 20 most frequently
## used words in the corpus - mobydick
#
def answer_three(nltk_text, num_freq_words):
    # print(f'answer_three, printing first 10 words from the text {type(nltk_text)}')
    
    freq_dist = nltk.FreqDist(nltk_text)
    return(freq_dist.most_common(num_freq_words))
    

#-------------------------- Question 4 --------------------------

#
## Find tokens that have:
##  1) length greater than 5 and 
##  2) frequency of more than 150
#
def answer_four(moby_tokens, frequency):
    # print(f'Tokens received: {moby_tokens}')
    words_length_greater_than_five_chars = []
    return_list = []
    for word in moby_tokens:
        if(len(word) > 5):
            words_length_greater_than_five_chars.append(word)
        else:
            continue
 
    # print(f'   ---- Words longer than 5 characters ----:')
    # pprint(words_length_greater_than_five_chars, indent=4)
    words_length_greater_than_five_chars_nltk_text =  nltk.Text(words_length_greater_than_five_chars)
    freq_dist = nltk.FreqDist(words_length_greater_than_five_chars_nltk_text)
    # print(f'freq_dist type is: {type(freq_dist)}')
    
    for (word,count) in freq_dist.items():
        print(f'{word} appeared: {count}')
        if(count > frequency):
            this_tuple = tuple([word,count])
            return_list.append(this_tuple)
        else:
            continue
    
    pprint(return_list, indent=3)
        
    # freq_dist_list = list(freq_dist)
    # print("\n\n ---- Printing converted frequency distro to list ---- ")
    # print(freq_dist_list)
    # one_fifty_words_length_greater_than_five_chars = answer_three(words_length_greater_than_five_chars, frequency)
    
    
    
    # print(f'Type of words_len_gt_five_one_fifty: {type(words_len_gt_five_one_fifty)}')
    # print(f'150 most apearing words of length 5: {one_fifty_most_popular}')
    # return(words_len_gt_five_one_fifty)


#-------------------------- Question 5 --------------------------


#
## Find the longest word in text_to_nltk_text and that word's length.
#
def answer_five(moby_tokens):
    # print(f'Tokens received: {moby_tokens}')
    
    longest_word = None
    iteration = 0
    longest_word_length = 0
    
    for word in moby_tokens:
        if(iteration == 0):
            longest_word  = word
        elif(len(word) > len(longest_word)):
            # print(f'Comparing {word} to {longest_word}')
            longest_word = word
            longest_word_length = len(longest_word)
        iteration += 1
        
    return(longest_word, longest_word_length)
        
    
    


#
## Read in moby.txt file into raw string
with open('moby.txt', 'r') as f:
    moby_raw = f.read()


#
## Tokenize the raw text    
moby_tokens = nltk.word_tokenize(moby_raw)
# print(f'Tokens in moby: {moby_tokens}')

#
## Convert to NLTK Text object - Its methods perform a variety of analyses 
## on the text’s contexts (e.g., counting, concordancing, collocation 
## discovery), and display the results

text_to_nltk_text = nltk.Text(moby_tokens)

#
## Example 1 - How many tokens (words and punctuation symbols) are in text_to_nltk_text?
#
total_tokens = example_one()
print(f'Number of tokens: {total_tokens}')
print("---------------------------------------------------------------------------")


#
## Example 2 - How many unique tokens (unique words and punctuation) 
## does text_to_nltk_text have?
## This function should return an integer.*
#
total_unique_tokens = example_two()
print(f'Number of unique tokens: {total_unique_tokens}')
print("---------------------------------------------------------------------------")



#
## Example 3
## After lemmatizing the verbs, how many unique tokens does text_to_nltk_text have?
## This function should return an integer.
#
total_unique_tokens_lemmatized_verbs = example_three()
print(f'Number of unique tokens after lemmatizing verbs: {total_unique_tokens_lemmatized_verbs}')
print("---------------------------------------------------------------------------")


#
## Question 1
##--------------------------------------------------------
## What is the lexical diversity of the given text input
##  (i.e. ratio of unique tokens to the total number of tokens)
## This function should return an integer.
#
print("\n 1) Getting answer for question one\n")
lexical_diversity = answer_one(total_unique_tokens, total_tokens)
print(f'\tLexical diversity: {lexical_diversity}')
print("---------------------------------------------------------------------------")



#
## Question 2
##--------------------------------------------------------
##  What percentage of tokens is 'whale'or 'Whale'?
print("\n 2) Getting answer for question two")
print("\n\t 1) What percentage of tokens is 'whale'or 'Whale'?\n")

whale_occurance = answer_two(text_to_nltk_text, "whale", moby_tokens)
whale_occurance_percentage = ((whale_occurance / total_tokens) * 100)
print(f'\tNumber of occurances of the word whale/Whale: {whale_occurance}')
print(f'\tOccurange of the word whale in percentage: {whale_occurance_percentage}')
print("---------------------------------------------------------------------------")



#
## Question 3
##--------------------------------------------------------
## What are the 20 most frequently occurring (unique) tokens in the 
## text? What is their frequency?
##
## This function should return a list of 20 tuples where each tuple is of 
## the form (token, frequency). The list should be sorted in descending 
## order of frequency.
##
## Resource:
##  https://www.nltk.org/book/ch01.html
##--------------------------------------------------------
print("\n 3) Getting answer for question three:")
print("\n\t   1) Twenty most frequently occurring (unique) tokens in the")
print("\n\t   2) Frequency of the twenty most frequenty occurring words\n")

num_freq_words = 20
twenty_most_common = answer_three(text_to_nltk_text, num_freq_words)

#
## Uncomment the next 2 lines
# print("Most commonly used words/tokens:")
# pprint(twenty_most_common, indent=4)
print("---------------------------------------------------------------------------")



#
## Question 4
##--------------------------------------------------------
## Find the tokens that have a length of greater than 5 
## and frequency of more than 150?
##
## Resource:
##  
##--------------------------------------------------------
print("\n\t 4) Getting answer for question four - Find the tokens that")
print("\n\t   1) Have a length of greater than 5")
print("\n\t   2) Frequency of more than 150\n")

## and frequency of more than 150")
word_frequency = 150
one_hundred_fiftey_words_with_length_five_plus = answer_four(text_to_nltk_text, word_frequency)
print("Words longer than 5 chars long and appearing at least 150 times:")
pprint(one_hundred_fiftey_words_with_length_five_plus, indent=2)

print("---------------------------------------------------------------------------")


