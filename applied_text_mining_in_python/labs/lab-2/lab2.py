
import collections
import nltk
import pandas as pd
from pprint import pprint
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.metrics.distance import jaccard_distance
import ssl


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
## Download nltk modules
#
def download_nltk_data(module_name, module_dir):
  nltk_data_dir = '/usr/local/share/nltk_data'
  nltk.data.path.append(nltk_data_dir)

  module_search_path = nltk_data_dir + '/' + module_dir + '/' + module_name
  print(f'\nChecking for module {module_name} in {module_search_path}')
  
  try:
    nltk.data.find(module_search_path)
  except LookupError:
    print(f'Module {module_name} not found, will attempt to download')
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download(module_name)

  # nltk.download('punkt')

  




#
## This subroutine will download necessary nltk modules
## Resource(s):
##  https://tinyurl.com/2p8wehdn
##
##
## nltk download directory can be configured both by command-line 
## (nltk.download(..., download_dir=) or by GUI
## /usr/local/share/nltk_data should be the central directory
## for nltk data
##
def nltk_data_downloader():

  from nltk.stem import WordNetLemmatizer
  from nltk.tokenize import sent_tokenize
    
  modules_dict = {
                   'wordnet' : 'corpora',
                   'omw-1.4' : 'corpora',
                   'stopwords':'corpora',
                   'words':'none',
                   'averaged_perceptron_tagger': 'taggers',
                   'punkt': 'tokenizers', # Provides word_tokenize and sent_tokenize
                 }
  
  for module in modules_dict:
    module_type = modules_dict[module]
    # print(module, module_type)
    download_nltk_data(module, module_type) 

  print("\n")
      


#
## Returns number of tokens (words and punctuation symbols) are in moby_nltk_text
#
def example_one(moby_word_tokens_punctuations_removed):
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(nltk_text)


#
## Returns number of tokens (words and punctuation symbols) are in nltk_text
#
def example_two():
    
    # return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(nltk_text))
    return (len(moby_word_tokens_punctuations_removed)) # or alternatively len(moby_nltk_text)


#
## Returns number of tokens (words and punctuation symbols) are in moby_nltk_text
#
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(moby_nltk_text))


#
## Returns number of unique tokens, after lemmatizing the verbs
#
def example_three():

    lemmatizer = WordNetLemmatizer()
    # lemmatized = [lemmatizer.lemmatize(w,'v') for w in nltk_text]
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in moby_nltk_text]

    return len(set(lemmatized))


#-------------------------- Answer 1 --------------------------


#
## Returns number of unique tokens, after lemmatizing the verbs
#
def answer_one(unique_tokens, total_tokens):    
    
    return(unique_tokens/total_tokens)
#------------------------------------------------------------------------------


#-------------------------- Answer 2 --------------------------
#
## Return the lexical diversity of the given text input
## Resource: https://www.nltk.org/howto/concordance.html
#
def answer_two(text, token, moby_word_tokens):    
    whale_count = 0
    
    # return(len(text.concordance_list(token)))
    
    for iter_token in moby_word_tokens: 
        if((iter_token == 'whale') or (iter_token == 'Whale')):
            whale_count += 1
    
    return(whale_count)
#------------------------------------------------------------------------------



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
#------------------------------------------------------------------------------



#-------------------------- Answer 4 --------------------------

#
## Find tokens that have:
##  1) length greater than 5 and 
##  2) frequency of more than 150
#
def answer_four(moby_word_tokens, frequency):
    # print(f'Tokens received: {moby_word_tokens}')
    words_length_greater_than_five_chars = []
    return_list = []
    for word in moby_word_tokens:
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
#------------------------------------------------------------------------------



#-------------------------- Answer 5 --------------------------

#
## Find the longest word in moby_nltk_text and that word's length.
#
def answer_five(moby_word_tokens):
    # print(f'Tokens received: {moby_word_tokens}')
    
    longest_word = None
    iteration = 0
    longest_word_length = 0
    
    for word in moby_word_tokens:
        if(iteration == 0):
            longest_word  = word
        elif(len(word) > len(longest_word)):
            # print(f'Comparing {word} to {longest_word}')
            longest_word = word
            longest_word_length = len(longest_word)
        iteration += 1
        
    return(longest_word, longest_word_length)
#------------------------------------------------------------------------------



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
#------------------------------------------------------------------------------


    

#-------------------------- Answer 7 --------------------------
## What is the average number of tokens per sentence?
## This function should return a float. 
#
#
def answer_seven(moby_sentence_tokens):
    counter = 0
    total_number_of_words_in_all_sentences = 0
    total_number_of_words_in_this_sentences = 0
    
    for sentence in moby_sentence_tokens:
        total_number_of_words_in_this_sentences = len(set(nltk.word_tokenize(sentence)))
        # print(f'  Sentence {counter} with length {sent_len} is: ')
        # pprint(sentence, indent=4)
        counter += 1
        total_number_of_words_in_all_sentences += total_number_of_words_in_this_sentences
    
    average_number_of_tokens_per_sentence = (total_number_of_words_in_all_sentences / counter)
    return(average_number_of_tokens_per_sentence)
#------------------------------------------------------------------------------


    

#-------------------------- Answer 8 --------------------------
## What are the 5 most frequent parts of speech in this text? 
## What is their frequency?
#
#  Reference(s):
#    https://www.youtube.com/watch?v=6j6M2MtEqi8&t=89s
#    https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
#
#
#  Remove all punctuation marks:
#    https://www.adamsmith.haus/python/answers/how-to-remove-all-punctuation-marks-with-nltk-in-python
#
#  CC coordinating conjunction
#  CD cardinal digit
#  DT determiner
#  EX existential there (like: "there is" ... think of it like "there exists")
#  FW foreign word
#  IN preposition/subordinating conjunction
#  JJ adjective 'big'
#  JJR adjective, comparative 'bigger'
#  JJS adjective, superlative 'biggest'
#  LS list marker 1)
#  MD modal could, will
#  NN noun, singular 'desk'
#  NNS noun plural 'desks'
#  NNP proper noun, singular 'Harrison'
#  NNPS proper noun, plural 'Americans'
#  PDT predeterminer 'all the kids'
#  POS possessive ending parent's
#  PRP personal pronoun I, he, she
#  PRP$ possessive pronoun my, his, hers
#  RB adverb very, silently,
#  RBR adverb, comparative better
#  RBS adverb, superlative best
#  RP particle give up
#  TO to go 'to' the store.
#  UH interjection errrrrrrrm
#  VB verb, base form take
#  VBD verb, past tense took
#  VBG verb, gerund/present participle taking
#  VBN verb, past participle taken
#  VBP verb, sing. present, non-3d take
#  VBZ verb, 3rd person sing. present takes
#  WDT wh-determiner which
#  WP wh-pronoun who, what
#  WP$ possessive wh-pronoun whose
#  WRB wh-abverb where, when
# 
#
## working solution
def answer_eight(moby_raw):
  
  # Tokenize the corpora
  tokens = nltk.word_tokenize(moby_raw)
    
  #Use part-of-speech to tagg.
  pos_tagged_list = nltk.pos_tag(tokens)    
  pos_counts = collections.Counter((key_val[1] for key_val in pos_tagged_list))
  # print(f'The five most common tags are {pos_counts.most_common(5)}') 
  return (pos_counts.most_common(5))     
#------------------------------------------------------------------------------


    

#-------------------------- Answer 9 --------------------------
## Create a spelling recommenders, that takes a list of misspelled words 
## and recommends a correctly spelled word for every word in the list.
#
#
#  Reference(s):
#
#   https://mariogarcia.github.io/blog/2021/04/nlp_text_similarity.html
#   https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/
#
## 

def get_ngrams(word, n):
    return set(ngrams(word,n=n))

def answer_nine(entries=['cormulent', 'incendenece', 'validrate'], gram_size=3, jacc_treshold=0.39):
    
  # Create an empty dictionary
  correct_words_list = {}
  
  # A threshold number, above that is like correct
  # valid_threshold = 0.39
  
  score = 0
  spellings = words.words()
  
  for bad_word in entries:
    # print(f'\n   << Bad word is: {bad_word} >>\n')
    
    bad_word_ngram = get_ngrams(bad_word, gram_size)
    sorted_correct_words_list = {}  
    
    for word in spellings:
      
      if(word.startswith(bad_word[0])):
        word_ngram = get_ngrams(word, gram_size)
        
        # print("   -------------------------------------")
        # print(f'   Comparing: {word} to {bad_word}')
        jacc_distance = (1 - jaccard_distance(set(bad_word_ngram), set(word_ngram)))
        # print(f'   Jaccard distance is: {jacc_distance}\n')
        
        # Store the word with its respective jaccard distance in a dictionary
        if(jacc_distance > jacc_treshold):
          correct_words_list[word] = jacc_distance
          # print("   -------------------------------------")
          # print(f'\n     << Suggestion - {bad_word} should be {word} >>\n')
          
        else:
          continue
    
      else:
        continue
        
  # sorted_correct_words_list = sorted(correct_words_list, key=correct_words_list.get)
  sorted_correct_words_list = sorted(correct_words_list.items(), key=lambda dict: dict[1], reverse=True)
  
  return(sorted_correct_words_list)
  
  # print(f'\n   Likely correct words are: {sorted_correct_words_list[0:5]}')
  # sorted_correct_words_list = {}
  
#------------------------------------------------------------------------------




#-------------------------- Answer 11 --------------------------
## Create a spelling recommenders, that takes a list of misspelled words 
## and recommends a correctly spelled word for every word in the list.
#
#
#  Reference(s):
#
#   https://tedboy.github.io/nlps/generated/generated/nltk.edit_distance.html
#
## 
def answer_eleven(entries=['cormulent', 'incendenece', 'validrate'], transpose='true'):
    
  # Get a list of words 
  spellings = words.words()
  
  # For each word in entries, iterate over earch word 
  for bad_word in entries:
    # print(f'\n   << Bad word is: {bad_word} >>\n')
    
    sorted_correct_words_list = {}  
    correct_words_list = {}
    
    # Iterate over each word from the spellings
    for word in spellings:
      
      # If word from spelling strats with the same letter as
      # bad_word
      if(word.startswith(bad_word[0]) and (len(word) == len(bad_word))):
        print(f'   Comparing: {word} to {bad_word}')
        levenshtein_distance = nltk.edit_distance(word, bad_word, transpositions=transpose)
        
        #
        # if levenshtein distance is less than 3, then add the word to the
        # corrected_words_list
        if(levenshtein_distance < 4):
        
          # print(f'Levenshtein distance is: {levenshtein_distance}')
          correct_words_list[word] = levenshtein_distance
          print(f'   Possible word matched is: {word} for {bad_word}')
        
        else:
          continue
        

      # If word from spelling does not strat with the same letter as
      # bad_word, then skip the loop
      else:
        continue
        
      print("\n-----------------------------------")

  # sorted_correct_words_list = sorted(correct_words_list, key=correct_words_list.get)
  sorted_correct_words_list = sorted(correct_words_list.items(), key=lambda dict: dict[1])
  return(sorted_correct_words_list[0:3])
  
#------------------------------------------------------------------------------

    
    
#
## Read in moby.txt file into raw string
with open('moby.txt', 'r') as f:
    moby_raw = f.read()


#
## Download nltk data/models, etc
nltk_data_downloader()

#
## Sentence Tokens
moby_sent_tokens = sent_tokenize(moby_raw)


#
## Tokenize the raw text    
moby_word_tokens = nltk.word_tokenize(moby_raw)
# print(f'Tokens in moby: {moby_word_tokens}')


#
## Tokenize and remove punctuations
punctuation_tokenizer = nltk.RegexpTokenizer(r"\w+")
moby_word_tokens_punctuations_removed = punctuation_tokenizer.tokenize(moby_raw)
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

nltk_text = nltk.Text(moby_word_tokens)
moby_nltk_text = nltk.Text(moby_word_tokens_punctuations_removed)

#
## Example 1 - How many tokens (words and punctuation symbols) are in moby_nltk_text?
#
total_tokens = example_one(moby_word_tokens_punctuations_removed)
print(f'Number of tokens: {total_tokens}')
print("---------------------------------------------------------------------------")


#
## Example 2 - How many unique tokens (unique words and punctuation) 
## does moby_nltk_text have?
## This function should return an integer.*
#
total_unique_tokens = example_two()
print(f'Number of unique tokens: {total_unique_tokens}')
print("---------------------------------------------------------------------------")



#
## Example 3
## After lemmatizing the verbs, how many unique tokens does moby_nltk_text have?
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

# whale_occurance = answer_two(nltk_text, "whale", moby_tokens)
whale_occurance = answer_two(moby_nltk_text, "whale", moby_word_tokens)
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
# twenty_most_common = answer_three(nltk_text, num_freq_words)
twenty_most_common = answer_three(moby_nltk_text, num_freq_words)
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
# one_hundred_fiftey_words_with_length_five_plus = answer_four(nltk_text, word_frequency)
one_hundred_fiftey_words_with_length_five_plus = answer_four(moby_nltk_text, word_frequency)
pprint(one_hundred_fiftey_words_with_length_five_plus, indent=4)

print("---------------------------------------------------------------------------")


#
## Question 5
##--------------------------------------------------------
## Find the longest word in moby_nltk_text and that word's length.
##
## Resource:
##  
##--------------------------------------------------------
print("\n 5) Getting answer for question five")
print("\n   1) Find the longest word in moby_nltk_text and that word's length.")
# longest_word, length = answer_five(moby_tokens)
longest_word, length = answer_five(moby_word_tokens)
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
# unique_freq_gt_two_thousand = answer_six(nltk_text)
print("\n 6) Getting answer for question six")
print("\n   1) Find unique words that have a frequency of more than 2000.")
print("\n   2) What is the frequency of those words\n")
asking_frequency = 2000
freq_distro = answer_six(moby_nltk_text, asking_frequency)
pprint(freq_distro, indent=4)
print("---------------------------------------------------------------------------")



#
## Question 7
##--------------------------------------------------------
## What is the average number of tokens per sentence?
## This function should return a float.
##
print("\n 7) Getting answer for question seven")
print("\n   1) What is the average number of tokens per sentence?")
avg_tokens_per_sent = answer_seven(moby_sent_tokens)
print(f'   Average number of tokens per sentence: {avg_tokens_per_sent}')



#
## Question 8
##--------------------------------------------------------
## What are the 5 most frequent parts of speech in this text? 
## What is their frequency?
print("\n 8) Getting answer for question eight")
print("\n   1) What are the 5 most frequent parts of speech in this text?")
five_most_frequent_pos = answer_eight(moby_raw)
pprint(five_most_frequent_pos, indent=4)


#
## Question 9
##--------------------------------------------------------
## For this recommender, your function should provide recommendations 
## for the three default words provided above using the following distance 
## metric:
##          Jaccard distance
####
print("\n 9) Getting correct spelling suggestions for:")
print("\n   cormulent, incendenece, validrate  with ngram=3")
this_gram_size = 3
this_jacc_threshold=0.31
answer_nine(entries=['cormulent', 'incendenece', 'validrate'], gram_size=this_gram_size, jacc_treshold=this_jacc_threshold)



#
## Question 10
##--------------------------------------------------------
## For this recommender, your function should provide recommendations 
## for the three default words provided above using the following distance 
## metric:
##         Jaccard distance
####
print("\n 10) Getting correct spelling suggestions for:")
print("\n   cormulent, incendenece, validrate with ngram=4") 
this_gram_size = 4
this_jacc_threshold=0.2
answer_nine(entries=['cormulent', 'incendenece', 'validrate'], gram_size=this_gram_size, jacc_treshold=this_jacc_threshold)


#
## Question 11
##--------------------------------------------------------
## For this recommender, your function should provide recommendations 
## for the three default words provided above using the following distance 
## metric:
##          Levenshtein distance with transposition
####
print("\n 11) Getting correct spelling suggestions for:")
print("\n   cormulent, incendenece, validrate\n") 
this_gram_size = 4
this_jacc_threshold=0.2
question_eleven = answer_eleven(entries=['cormulent', 'incendenece', 'validrate'], transpose='true')
print(f'\n   Answer to question 11: {question_eleven}')
