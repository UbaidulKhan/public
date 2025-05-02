

import nltk
import pandas as pd
from pprint import pprint
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

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
<<<<<<< HEAD
## Returns number of tokens (words and punctuation symbols) are in nltk_text
=======
## Returns number of tokens (words and punctuation symbols) are in moby_nltk_text
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
#
def example_one(moby_word_tokesn_punctuations_removed):
    
<<<<<<< HEAD
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(nltk_text)


#
## Returns number of tokens (words and punctuation symbols) are in nltk_text
#
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(nltk_text))
=======
    return (len(moby_word_tokesn_punctuations_removed)) # or alternatively len(moby_nltk_text)


#
## Returns number of tokens (words and punctuation symbols) are in moby_nltk_text
#
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(moby_nltk_text))
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95


#
## Returns number of unique tokens, after lemmatizing the verbs
#
def example_three():

    lemmatizer = WordNetLemmatizer()
<<<<<<< HEAD
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in nltk_text]
=======
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in moby_nltk_text]
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95

    return len(set(lemmatized))


#-------------------------- Answer 1 --------------------------


#
## Returns number of unique tokens, after lemmatizing the verbs
#
def answer_one(unique_tokens, total_tokens):    
    
    return(unique_tokens/total_tokens)


#-------------------------- Answer 2 --------------------------


#
## Return the lexical diversity of the given text input
## Resource: https://www.nltk.org/howto/concordance.html
#
def answer_two(text, token, moby_word_tokesn):    
    whale_count = 0
    
    # return(len(text.concordance_list(token)))
    
    for iter_token in moby_word_tokesn: 
        if((iter_token == 'whale') or (iter_token == 'Whale')):
            whale_count += 1
    
    return(whale_count)


#-------------------------- Answer 3 --------------------------


#
## Return a tuple(word, occurance) of the 20 most frequently
## used words in the corpus - mobydick
#
def answer_three(text, num_freq_words):
    # print(f'answer_three, printing first 10 words from the text {text[10]}')
    # pprint(text[10], indent=4)
    freq_dist = nltk.FreqDist(text)
    return(freq_dist.most_common(num_freq_words))
    

#-------------------------- Answer 4 --------------------------

#
## Find tokens that have:
##  1) length greater than 5 and 
##  2) frequency of more than 150
#
def answer_four(moby_word_tokesn, frequency):
    # print(f'Tokens received: {moby_word_tokesn}')
    words_length_greater_than_five_chars = []
    return_list = []
    for word in moby_word_tokesn:
        if(len(word) > 5):
            words_length_greater_than_five_chars.append(word)
        else:
            continue
 
    #
    ## Convert the array words_length_greater_than_five_chars to nltk.Text
    #
    words_length_greater_than_five_chars_nltk_text =  nltk.Text(words_length_greater_than_five_chars)

    #
    ## Compute the frequency distribution on words_length_greater_than_five_chars_nltk_text
    freq_dist = nltk.FreqDist(words_length_greater_than_five_chars_nltk_text)
    # print(f'freq_dist type is: {type(freq_dist)}')
    
    #
    ## Loop through each item(word and frequency), if the frequency is greater
    ## than the frequency passed into the sub, then create a tuple from the 
    ## word and its frequency. Then append that tuple to the return_list
    for (word,count) in freq_dist.items():
        # print(f'{word} appeared: {count}')
        if(count > frequency):
            this_tuple = tuple([word,count])
            return_list.append(this_tuple)
        else:
            continue
    
    # pprint(return_list, indent=3)
    return(return_list)


#-------------------------- Answer 5 --------------------------

#
<<<<<<< HEAD
## Find the longest word in nltk_text and that word's length.
=======
## Find the longest word in moby_nltk_text and that word's length.
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
#
def answer_five(moby_word_tokesn):
    # print(f'Tokens received: {moby_word_tokesn}')
    
    longest_word = None
    iteration = 0
    longest_word_length = 0
    
    for word in moby_word_tokesn:
        if(iteration == 0):
            longest_word  = word
        elif(len(word) > len(longest_word)):
            # print(f'Comparing {word} to {longest_word}')
            longest_word = word
            longest_word_length = len(longest_word)
        iteration += 1
        
    return(longest_word, longest_word_length)


#-------------------------- Answer 6 --------------------------
## Find unique words with a frequency of 2000 or more than? 
## What is their frequency?
#
#
def answer_six(moby_nltk_text, frequency):
    #
    ## Compute the frequency distribution on words_length_greater_than_five_chars_nltk_text
    freq_dist = nltk.FreqDist(moby_nltk_text)
    # print(f'freq_dist type is: {type(freq_dist)}')
    
    #
    ## A return list that will be populated with tokens and their frequency
    return_list = []
    
    #
    ## Loop through each item(word and frequency), if the frequency is greater
    ## than the frequency passed into the sub, then create a tuple from the 
    ## word and its frequency. Then append that tuple to the return_list
    for (word,count) in freq_dist.items():
        # print(f'{word} appeared: {count}')
        if(count > frequency):
            this_tuple = tuple([word,count])
            return_list.append(this_tuple)
        else:
            continue
    
    # pprint(return_list, indent=3)
    return(return_list)
    

#-------------------------- Answer 7 --------------------------
## What is the average number of tokens per sentence?
## This function should return a float. 
#
#
def answer_seven(moby_sentence_tokens):
    counter = 0
    for sentence in moby_sentence_tokens:
        sent_len = len(set(nltk.word_tokenize(sentence)))
        print(f'  Sentence {counter} with length {sent_len} is: ')
        pprint(sentence, indent=4)
        counter += 1
    
            
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<===>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#
## Read in moby.txt file into raw string
with open('moby.txt', 'r') as f:
    moby_raw = f.read()


#
## Setnace Tokens
moby_sent_tokens = sent_tokenize(moby_raw)


#
## Tokenize the raw text    
moby_word_tokesn = nltk.word_tokenize(moby_raw)
# print(f'Tokens in moby: {moby_word_tokesn}')


#
## Tokenize and remove punctuations
punctuation_tokenizer = nltk.RegexpTokenizer(r"\w+")
moby_word_tokesn_punctuations_removed = punctuation_tokenizer.tokenize(moby_raw)
# print(f'Words with punctuation removed: {moby_words_punctuations_removed }')


#
## NOTE - tokenizer that can remove punctuation
## use "RegexpTokenizer" to tokenize only word without punctuations
## tokenizer = nltk.RegexpTokenizer(r"\w+")
## new_words = tokenizer.tokenize(sentence)

#
## nltk.Text is a wrapper around a sequence of simple (string) tokens, which is 
## intended to support initial exploration of texts (via the interactive console). 
## Its methods perform a variety of analyses on the text’s contexts (e.g., counting, 
## concordancing, collocation discovery), and display the results

<<<<<<< HEAD
nltk_text = nltk.Text(moby_tokens)

#
## Example 1 - How many tokens (words and punctuation symbols) are in nltk_text?
=======
moby_nltk_text = nltk.Text(moby_word_tokesn_punctuations_removed)

#
## Example 1 - How many tokens (words and punctuation symbols) are in moby_nltk_text?
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
#
total_tokens = example_one(moby_word_tokesn_punctuations_removed)
print(f'Number of tokens: {total_tokens}')
print("---------------------------------------------------------------------------")


#
## Example 2 - How many unique tokens (unique words and punctuation) 
<<<<<<< HEAD
## does nltk_text have?
=======
## does moby_nltk_text have?
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
## This function should return an integer.*
#
total_unique_tokens = example_two()
print(f'Number of unique tokens: {total_unique_tokens}')
print("---------------------------------------------------------------------------")



#
## Example 3
<<<<<<< HEAD
## After lemmatizing the verbs, how many unique tokens does nltk_text have?
=======
## After lemmatizing the verbs, how many unique tokens does moby_nltk_text have?
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
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
print(f'    Lexical diversity: {lexical_diversity}')
print("---------------------------------------------------------------------------")



#
## Question 2
##--------------------------------------------------------
##  What percentage of tokens is 'whale'or 'Whale'?
print("\n 2) Getting answer for question two")
print("\n 1) What percentage of tokens is 'whale'or 'Whale'?\n")

<<<<<<< HEAD
whale_occurance = answer_two(nltk_text, "whale", moby_tokens)
=======
whale_occurance = answer_two(moby_nltk_text, "whale", moby_word_tokesn)
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
whale_occurance_percentage = ((whale_occurance / total_tokens) * 100)
print(f'    Number of occurances of the word whale/Whale: {whale_occurance}')
print(f'    Occurange of the word whale in percentage: {whale_occurance_percentage}')
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
print("\n   1) Twenty most frequently occurring (unique) tokens in the")
print("\n   2) Frequency of the twenty most frequenty occurring words\n")

num_freq_words = 20
<<<<<<< HEAD
twenty_most_common = answer_three(nltk_text, num_freq_words)
=======
twenty_most_common = answer_three(moby_nltk_text, num_freq_words)
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
pprint(twenty_most_common, indent=4)
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
print("\n 4) Getting answer for question four - Find the tokens that")
print("\n   1) Have a length of greater than 5")
print("\n   2) Frequency of more than 150\n\t")

## and frequency of more than 150")
word_frequency = 150
<<<<<<< HEAD
one_hundred_fiftey_words_with_length_five_plus = answer_four(nltk_text, word_frequency)
=======
one_hundred_fiftey_words_with_length_five_plus = answer_four(moby_nltk_text, word_frequency)
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
pprint(one_hundred_fiftey_words_with_length_five_plus, indent=4)

print("---------------------------------------------------------------------------")


#
## Question 5
##--------------------------------------------------------
<<<<<<< HEAD
## Find the longest word in nltk_text and that word's length.
=======
## Find the longest word in moby_nltk_text and that word's length.
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
##
## Resource:
##  
##--------------------------------------------------------
print("\n 5) Getting answer for question five")
<<<<<<< HEAD
print("\n   1) Find the longest word in nltk_text and that word's length.")
longest_word, length = answer_five(moby_tokens)
=======
print("\n   1) Find the longest word in moby_nltk_text and that word's length.")
longest_word, length = answer_five(moby_word_tokesn)
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95
print(f'      Longest word is: {longest_word} of the length: {length}')
print("---------------------------------------------------------------------------")


#
## Question 6
##--------------------------------------------------------
## What unique words have a frequency of more than 2000? 
## What is their frequency?
##
## Resource:
##   How to remove punctuations:
##
##   https://www.adamsmith.haus/python/answers/how-to-remove-all-punctuation-marks-with-nltk-in-python
##
##   sentence  = "Think and wonder, wonder and think."
##   
##   tokenizer = nltk.RegexpTokenizer(r"\w+")
##   new_words = tokenizer.tokenize(sentence)
##  
##--------------------------------------------------------
<<<<<<< HEAD
unique_freq_gt_two_thousand = answer_six(nltk_text)
=======
print("\n 6) Getting answer for question six")
print("\n   1) Find unique words have a frequency of more than 2000.")
print("\n   2) What is the frequency of those words\n")
asking_frequency = 2000
freq_distro = answer_six(moby_nltk_text, asking_frequency)
pprint(freq_distro, indent=4)
print("---------------------------------------------------------------------------")
>>>>>>> 60cda93aef300361a804846517225e11e40fdd95



#
## Question 7
##--------------------------------------------------------
## What is the average number of tokens per sentence?
## This function should return a float.
##
avg_tokens_per_sent = answer_seven(moby_sent_tokens)