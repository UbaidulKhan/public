#
#
##  Copyright(c), 2023
##  Ubaidul Khan, ubaidul.khan@gmail.com
##
##  File-name: lab4p1.py, Lab 4 part 1.
##
##  Meta: 
##   Applied Text Mining in Python, lab-4
##   University of Michigan, Coursera
##
## Lab URL:
## https://www.coursera.org/learn/python-text-mining/programming/2qbcK/assignment-4/instructions
##
##
#
## Additional Resource/Solution:
##  https://github.com/umer7/Applied-Text-Mining-in-Python/blob/master/Assignment%2B4.ipynb
##
#
## To Read:
#
#  Sample usage for wordnet
#
#   https://www.nltk.org/howto/wordnet.html
#
#  Setndex Wordnet with NLTK
# 
#   https://pythonprogramming.net/wordnet-nltk-tutorial/
#
#  Parts of Speech Tagging:
#
#   https://pythonexamples.org/nltk-pos-tagging/
#
#
#
import inspect
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import os
import pandas as pd
from pandas import option_context
import re
from tabulate import tabulate
#
## To generate call-graph run:
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#
## Debugging flag - when true, it will produce debugging output
# debug_flag = True
debug_flag = False

# +---------------------------------------------------------------------------+
#                               DATA AND STRUCTURE
# +---------------------------------------------------------------------------+
#
## paraphases.csv is the file that will be used in this lab - this file contains
## 3 columns:
##
#    1) Quality
#    2) D1      - document 1
#    3) D2      - document 2
##
## Here is an example:
#
# +----------------------------------------------+---------------------------+
# |  QUALITY  |    D1(document 1)                |   D1(document 2)          |
# +--------------------------------------------------------------------------+
# |  1        |  Druce will face murder charges, |  Conte said Druce will be |          
# |           |  Conte said.                     |  charged with murder      |
# +----------------------------------------------+---------------------------+
#
# 
# +---------------------------------------------------------------------------+
#                               TASKS
# +---------------------------------------------------------------------------+
#
## Part 1:
#
#  - test_document_path_similarity 
#  - Document Similarity
# 
#------------------------------------------------------------------------------
# For the first part of this assignment, you will complete the functions: 
#
#################################### TASKS ####################################
#
#  >> doc_to_synsets 
#  >> similarity_score 
#
#
#################################### GIVEN ####################################
#
# 
# The following functions are provided:
# 
# convert_tag: converts the tag given by nltk.pos_tag to a tag used by 
# wordnet.synsets. You will need to use this function in doc_to_synsets.
# 
# document_path_similarity: computes the symmetrical path similarity between 
# two documents by finding the synsets in each document using doc_to_synsets, 
# then computing similarities using similarity_score.
# 
#
#################################### TASKS ####################################
#
# You will need to finish writing the following functions:
#
#  >> doc_to_synsets 
#  >> similarity_score 
#
# which will be used by document_path_similarity to find the path similarity 
# between two documents.
# 
# doc_to_synsets: returns a list of synsets in document. This function should 
# first tokenize and part of speech tag the document using nltk.word_tokenize 
# and nltk.pos_tag. Then it should find each tokens corresponding synset using 
# wn.synsets(token, wordnet_tag). The first synset match should be used. If 
# there is no match, that token is skipped.
# 
# similarity_score: returns the normalized similarity score of a list of 
# synsets(s1) onto a second list of synsets (s2). For each synset in s1, find 
# the synset in s2 with the largest similarity value. Sum all of the largest 
# similarity values together and normalize this value by dividing it by the 
# number of largest similarity values found. Be careful with data types, which 
# should be floats. Missing values should be ignored.
# 
# Once doc_to_synsets and similarity_score have been completed, submit to the 
# autograder which will run test_document_path_similarity to test that these 
# functions are running correctly.



#
## ANCILIRY: pos_tag_lookup()
##-----------------------------------------------------------------------------
#
# Provide function takes a parts of speech tag - CC, CD, DT, NN, etc
#
# Returns the expanded form of the tag
#
## References:
##
##  https://www.guru99.com/pos-tagging-chunking-nltk.html
##
## To do: 
##  1) 
#
# mfnt - method acronym
# 
def pos_tag_lookup(tag):
  
  pos_table_lookup = {
      'CC':   'Coordinating Conjunction',
      'CD':   'Cardinal Digit',
      'DT':   'Determiner',
      'EX':   'Existential There. Example: “there is” … think of it like “there exists”)',
      'FW':   'Foreign Word.',
      'IN':   'Preposition/Subordinating Conjunction.',
      'JJ':   'Adjective.',
      'JJR':  'Adjective, Comparative.',
      'JJS':  'Adjective, Superlative.',
      'LS':   'List Marker 1.',
      'MD':   'Modal.',
      'NN':   'Noun, Singular.',
      'NNS':  'Noun Plural.',
      'NNP':  'Proper Noun, Singular.',
      'NNPS': 'Proper Noun, Plural.',
      'PDT':  'Predeterminer.',
      'POS':  'Possessive Ending. Example: parent’s',
      'PRP':  'Personal Pronoun. Examples: I, he, she',
      'PRP$': 'Possessive Pronoun. Examples: my, his, hers',
      'RB':   'Adverb. Examples: very, silently,',
      'RBR':  'Adverb, Comparative. Example: better',
      'RBS':  'Adverb, Superlative. Example: best',
      'RP':   'Particle. Example: give up',
      'TO':   'to. Example: go ‘to’ the store.',
      'UH':   'Interjection. Example: errrrrrrrm',
      'VB':   'Verb, Base Form. Example: take',
      'VBD':  'Verb, Past Tense. Example: took',
      'VBG':  'Verb, Gerund/Present Participle. Example: taking',
      'VBN':  'Verb, Past Participle. Example: taken',
      'VBP':  'Verb, Sing Present, non-3d take',
      'VBZ':  'Verb, 3rd person sing. present takes',
      'WDT':  'wh-determiner. Example: which',
      'WP':   'wh-pronoun. Example: who, what',
      'WP$':  'possessive wh-pronoun. Example: whose',
      'WRB':  'wh-abverb. Example: where, when'
  }
  
  expanded_tag = pos_table_lookup.get(tag, 'Unknown')
  return(expanded_tag)
    


#
## ANCILIRY: function_name_table_lookup()
##-----------------------------------------------------------------------------
#
# Provide function takes a function-name, then lookups up the function-name
# acronym from the func_name_dict.  
#
# Returns the acronym
#
## References:
##
##  https://www.geeksforgeeks.org/add-a-keyvalue-pair-to-dictionary-in-python/
##
## To do: 
##  1) Construct table
##  2) Write a look up function
#
# mfnt - method acronym
# 
def function_name_table_lookup(name):
  try:
    return(func_name_dict[name])
  except KeyError:
    return None
  


#
## ANCILIRY: dbg_dbg_printt()
##-----------------------------------------------------------------------------
#
# Provides debugging dbg_printting - when debug flag is set, it will dbg_printt
##  
def dbg_print(arguments):
  if(debug_flag):
    print(arguments)
  else:
    return



#
## ANCILIRY: make_function_name_table()
##-----------------------------------------------------------------------------
#
# Provide function takes a function-name and generates an acronym for the 
# function. Then this acronym is added to a dictionary.
#
## References:
##
##  https://www.geeksforgeeks.org/add-a-keyvalue-pair-to-dictionary-in-python/
##
## To do: 
##  1) Construct table
##  2) Write a look up function
#
# mfnt - method acronym
# 
def make_function_name_table(name):
  
  
  # my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_printt(" ")
  '''
  dbg_printt("\n")
  dbg_printt("+------------------------------------------------------------")
  dbg_printt(f'|   - make_function_name_table - ')
  dbg_printt("+------------------------------------------------------------")
  '''
  
  #
  ## Take the name received and split it by underscore(_)
  name_split = name.split('_')

  #dbg_printt(f'|  \_(ld) Name received: {name} ')
  #dbg_printt(f'|  \_(ld) Name Split: {name_split} ')

  acronym = ""
  word_pos = 0
  for word in name_split:
    #dbg_printt(f'|  \_(ld) Word in {word_pos} the : {word} ')
    #dbg_printt(f'|  \_(ld) First letter in word is {word[0]}')
    acronym = acronym + word[0]
    word_pos += 1
      
  #func_name_dict = {'make_function_name_table':'mfnt', }
  func_name_dict[name] = acronym
  
  # dbg_printt(f'  \_(ld) Function name table: {func_name_dict}')



#
## ANCILIRY: stack_inspector()
##-----------------------------------------------------------------------------
# 
#
def stack_inspector():
  
  curframe = inspect.currentframe()
  calframe = inspect.getouterframes(curframe, 2)
  
  # dbg_print(f'  Call frame length is: {len(calframe)}')
  
  my_name = my_parents_name = my_grand_parents_name = my_great_grand_parent = None

  #
  ## Determine a name of the current function name & acronym
  try:
    my_name = curframe.f_code.co_name
    make_function_name_table(my_name)
  except ValueError:
    dbg_print(f"Failed to read from file: {file_path}")
    
  #
  ## Determine a name of the parent function name & acronym
  if(calframe[1][3]):
    my_parents_name = calframe[1][3]
    make_function_name_table(my_parents_name)
 
    
  #
  ## Determine a name of the grand parent function name & acronym
  if(calframe[2][3]):
    my_grand_parents_name = calframe[2][3]
    make_function_name_table(my_grand_parents_name)
 
  '''   
  #
  ## Determine a name of the great grand parent function name & acronym
  try:
    if(calframe[3][3]):
      my_great_grand_parent = calframe[3][3]
      make_function_name_table(my_great_grand_parent)
      
  except ValueError:
    dbg_print(f"Failed calculate great grand parents: {ValueError}")
 
  '''   
    
  # dbg_print(f'{my_parents_name} , {my_grand_parents_name} , {my_great_grand_parent}')
  
  return(my_parents_name, my_grand_parents_name, my_great_grand_parent)

  # return(my_parents_name, my_grand_parents_name)
  


#
## TASK: similarity_score
#------------------------------------------------------------------------------
# similarity_score: returns the normalized similarity score of a list of 
# synsets (s1) onto a second list of synsets (s2). For each synset in s1, find
# the synset in s2 with the largest similarity value. Sum all of the largest 
# similarity values together and normalize this value by dividing it by the 
# number of largest similarity values found. Be careful with data types, which 
# should be floats. Missing values should be ignored.
#------------------------------------------------------------------------------
#
## Reference:
#    https://youtu.be/T68P5-8tM-Y?t=505 (sentdex)
#
def similarity_score(s1, s2):
  
  """
  Calculate the normalized similarity score of s1 onto s2

  For each synset in s1, finds the synset in s2 with the largest similarity value.
  Sum of all of the largest similarity values and normalize this value by dividing it by the
  number of largest similarity values found.

  Args:
    s1, s2: list of synsets from doc_to_synsets

  Returns:
    normalized similarity score of s1 onto s2

  Example:
    synsets1 = doc_to_synsets('I like cats')
    synsets2 = doc_to_synsets('I like dogs')
    similarity_score(synsets1, synsets2)
    Out: 0.73333333333333339
  """
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")
  
  largest_similarity_values = []
  for syn1 in s1:
    similarity_values = []
    for syn2 in s2: 
      sim_val = wn.path_similarity(syn1, syn2)
      if sim_val != None:
        similarity_values.append(sim_val)
    if len(similarity_values) != 0:
      largest_similarity_values.append(max(similarity_values))
  return sum(largest_similarity_values)/len(largest_similarity_values)



#
## GIVEN // DO NO MODIFY: document_path_similarity
##-----------------------------------------------------------------------------
# document_path_similarity: computes the symmetrical path similarity between 
# two documents by finding the synsets in each document using doc_to_synsets,
# then computing similarities using similarity_score
#------------------------------------------------------------------------------
#
def document_path_similarity(doc1, doc2, english_stopwords):
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")
  
  """Finds the symmetrical similarity between doc1 and doc2"""

  synsets1 = doc_to_synsets(doc1, english_stopwords)
  synsets2 = doc_to_synsets(doc2, english_stopwords)

  return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
    


#
## GIVEN // DO NOT MODIFY: test_document_path_similarity
##-----------------------------------------------------------------------------
# Using document_path_similarity, find the pair of documents in docs_from_csv 
# which has the maximum similarity score.
#
# This function should return a tuple (D1, D2, similarity_score)
#-----------------------------------------------------------------------------
# 
# def test_document_path_similarity(english_stopwords):
def test_document_path_similarity(english_stopwords):
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")
  
  doc1 = 'This is a function to test document_path_similarity.'
  doc2 = 'Use this function to see if your code in doc_to_synsets \
  and similarity_score is correct!'
  return document_path_similarity(doc1, doc2, english_stopwords)
    


#
## GIVEN: most_similar_docs
##-----------------------------------------------------------------------------
# Using document_path_similarity, find the pair of documents in docs_from_csv 
# which has the maximum similarity score.
#
# This function should return a tuple (D1, D2, similarity_score)
#-----------------------------------------------------------------------------
# 
def most_similar_docs():

  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")
  
  
  similarities = [(paraphrase['D1'], paraphrase['D2'], document_path_similarity(paraphrase['D1'], paraphrase['D2']))
                    for index, paraphrase in docs_from_csv.iterrows()]
  similarity = max(similarities, key=lambda item:item[2])
    
  return similarity
    


#
## GIVEN: label_accuracy
##-----------------------------------------------------------------------------
# Provide labels for the twenty pairs of documents by computing the similarity 
# for each pair using document_path_similarity. Let the classifier rule be that
# if the score is greater than 0.75, label is paraphrase (1), else label is not 
# paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.
#
# This function should return a float.
#-----------------------------------------------------------------------------
# 
def label_accuracy():
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  # dbg_print("\n")
  # dbg_print("+------------------------------------------------------------")
  # dbg_print(f'|   - {my_grand_parents_name} / {my_parents_name} - ')
  # dbg_print("+------------------------------------------------------------")
  
  from sklearn.metrics import accuracy_score
  df = docs_from_csv.apply(update_sim_score, axis=1)
  score = accuracy_score(df['Quality'].tolist(), df['paraphrase'].tolist())
    
  return score



#
## 
##-----------------------------------------------------------------------------
# 
#
# 
#-----------------------------------------------------------------------------
# 
def update_sim_score(row):
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  # dbg_print("\n")
  # dbg_print("+------------------------------------------------------------")
  # dbg_print(f'|   - {my_grand_parents_name} / {my_parents_name} - ')
  # dbg_print("+------------------------------------------------------------")
  
  row['similarity_score'] = document_path_similarity(row['D1'], row['D2'])
  row['paraphrase'] = 1 if row['similarity_score'] > 0.75 else 0
  return row



# 
#
################################ DO NOT MODIFY ################################
#
# Do not modify the functions:
#
#  >> convert_tag 
#  >> document_path_similarity
#  >> test_document_path_similarity
#
#
## GIVEN // DO NO MODIFY: convert_tag
#
#------------------------------------------------------------------------------
# convert_tag: converts the tag given by nltk.pos_tag to a tag used by 
# wordnet.synsets. You will need to use this function in doc_to_synsets.
#------------------------------------------------------------------------------
#
#
# ct - method acronym
# 
def convert_tag(token, tag):
  """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")

  dbg_print(f" | \_(ct) Token: *{token}* Tag received: *{tag}*")  
      
  # tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
  tag_dict = {'NNP': 'n', 'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}

  try:
    return_tag = (tag_dict[tag[0]])
    dbg_print(f" | \_(ct) Tag to return: *{return_tag}*") 

    dbg_print(" |............................................................")

    return(return_tag)
    
  except KeyError:
    return None



#
## TASK: doc_to_synsets
#------------------------------------------------------------------------------
# doc_to_synsets: returns a list of synsets in document. 
#
# A synset is a set of one or more synonyms that are interchangeable in some 
# context without changing the truth value of the proposition in which they are
# embedded.
#
# This function should first tokenize and part of speech tag the document using 
# nltk.word_tokenize and nltk.pos_tag. Then it should find each tokens 
# corresponding synset using wn.synsets(token, wordnet_tag). The first synset 
# match should be used. If there is no match, that token is skipped.
#------------------------------------------------------------------------------
#
## References:
#   https://pythonprogramming.net/wordnet-nltk-tutorial/
#   https://www.nltk.org/api/nltk.tag.pos_tag.html 
#   https://www.guru99.com/pos-tagging-chunking-nltk.html
#
# dts - method acronym
# 
def doc_to_synsets(doc, english_stopwords):
  """
  Returns a list of synsets in document.

  Tokenizes and tags the words in the document doc.
  Then finds the first synset for each word/tag combination.
  If a synset is not found for that combination it is skipped.

  Args:
    doc: string to be converted

  Returns:
    list of synsets

  Example:
    doc_to_synsets('Fish are nvqjp friends.')
    Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
  """
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")

  dbg_print(f" | \_(dts) Doc received:\n | \_ >> {doc} <<")  
  # dbg_print(f" | \_(dts) Stop words received:\n | \_ >> {english_stopwords} <<")  

  #
  ##!!!!!!!!!!!!!!!!!!!!!!!!!! REMOVE PUNCTUATION !!!!!!!!!!!!!!!!!!!!!!!!!!
  #
  doc = re.sub(r'[^\w\s]', '', doc)
  

  #
  ##!!!!!!!!!!!!!!!!!!!!!!!!!! REMOVE NEW LINE !!!!!!!!!!!!!!!!!!!!!!!!!!
  #
  doc = doc.strip()
  
  # dbg_print(f" | \_(dts) Doc without punctuation:")
  # dbg_print(f" | \_(dts) {doc}")
  #
  # This turned MS. into MS, which led to synset that included multiple_sclerosis


  #
  ## Create a data-frame with the doc as the column heading
  pos_df = pd.DataFrame(columns=[doc])
  # dbg_print(pos_df)
  
  #
  ## Create a parts-of-speech(POS) list for populating pos_df - data will be
  ## stored in the following form:
  ##
  ##  [ (token,
  ##     pos_tag, 
  ##     tag_summary)
  ##  ]
  ##
  df_entry_list = []
  
  #
  ##!!!!!!!!!!!!!!!!!!!!!!!!!! CONVERT TO LOWER CASE !!!!!!!!!!!!!!!!!!!!!!!!!!
  ## Side-Effect - this will convert certain nouns to verbs such as Stewart
  #
  # tokens = nltk.word_tokenize(doc.lower())
  # dbg_print(f" | \_(dts) tokens lower-case:")
  # dbg_print(f" | \_(dts) {tokens}")

  #
  ##!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CONVERT TO LOWER CASE !!!!!!!!!!!!!!!!!!!!!!!!!!
  #
  tokens = nltk.word_tokenize(doc)
  dbg_print(f" | \_(dts) tokens lower-case:")
  dbg_print(f" | \_(dts) {tokens}")


  #
  ##!!!!!!!!!!!!!!!!!!!!!!!!!! REMOVE STOP WORDS !!!!!!!!!!!!!!!!!!!!!!!!!!
  #
  ## !!! THIS IS NOT A GOOD IDEA - it removed 'not' !!!
  #
  # tokens_no_stopwords = [t for t in tokens if t not in english_stopwords]
  # dbg_print(f" | \_(dts) Doc without stop words: ** {tokens_no_stopwords} **")
  # tokens = [t for t in tokens if t not in english_stopwords]
  #
  #
  # Need to remove stop word such as MS - this led to MS having synset that
  # resolved to Synset('multiple_sclerosis.n.01') 
  #

  
  #
  ## NLTK Parts of Speech is used to tag the word tokens - nltk.pos_tag returns
  ## a tuple, similar to: [('John', 'NNP'), ('ate', 'VB')]
  # word_tags = nltk.pos_tag(tokens)
  word_tags = nltk.pos_tag(tokens)
  
  dbg_print(f" | \_(dts)")
  dbg_print(f" | \_(dts) Words/(tokens) with POS tags: \n | \_(dts)  ** {word_tags} **")

  #
  ## Construct a list for storing synonym sets
  synsets = []
  
  #
  ## The following is a counter that tracks the size of the data-frame that will
  ## be used for storing synset information
  ##
  df_row_index = 0
  
  for word, tag in word_tags:
    dbg_print(f" |.......................................................................... ")
    dbg_print(f" | \_(dts) Processing:")
    dbg_print(f" | \_(dts) < ** {word} ** : ** {tag} **>")
    dbg_print(f" | \_(dts) < ** {tag} ** expanded form: ** {pos_tag_lookup(tag)} ** >")
    
    #
    ## We need a string that will be used to generate a row in the data-frame:
    ## new_df_row. This string will contain the word, pos_tag and pos_tab
    ## explanation. This string will then become the value in a key/val dict
    ## the key is the doc
    ##
    #
    df_entry_string = str(word) + ' | ' + str(tag) + ' | ' + str(pos_tag_lookup(tag))
    dbg_print(f" | \_(dts) df entry: {df_entry_string}")
    
    #
    ## Convert tag - call convert_tag to get the first tag
    tag = convert_tag(word, tag)
    dbg_print(f" | \_(dts) Processing word ** {word} ** with converted Tag: ** {tag} **")
    
    #
    ## Find the sysnset for the word with the given post tag - Synset instances are the 
    ## groupings of synonymous words that express the same concept
    synset = wn.synsets(word, pos=tag)
    
    #
    ## If we received a set of synsets, then we need to grab the first synset
    ## and construct a data-frame-row with the synset.
    ##
    if (len(synset) != 0):
      synset_to_store = synset[0]
      synsets.append(synset_to_store)
      dbg_print(f" | \_(dts) Processing word: ** {word} **")
      dbg_print(f" | \_(dts) For tag: ** {tag} **, synsets: ** {synset_to_store} **")
      
      #
      ## Add the synset to df_entry_string after converting it to string
      df_entry_string = df_entry_string + ' | ' + str(synset_to_store)
      df_entry_string = df_entry_string.strip()
      
      #
      ## Create a dictionary 
      df_entry_dict = {doc:df_entry_string}
      dbg_print(f" | \_(dts) Data-framw row ** {df_entry_dict} **")
      
      #
      ## Create a data-frame using the dictionary 
      new_df_row = pd.DataFrame(df_entry_dict, index=[df_row_index])
      dbg_print(f" | \_(dts) Data-framw row ** {new_df_row} **")

      
      #
      ## Create a new list using the existing data-frame and the new
      ## data-frame(new_df_row)
      frames = [pos_df, new_df_row]

      pos_df = pd.concat(frames)
    
      #
      ## Increment the counter for the data-frame row count
      df_row_index = df_row_index + 1
      
      #
      ## Reset df_entry_string to empty
      df_entry_string = ''
      

    else:
      continue
  
  # dbg_print(f" | \_(dts) synset constructed: \n\n    ** {synsets} **")
  dbg_print(f" | \_(dts) Synset data-frame:")
  dbg_print(pos_df.head(3))
  
  dbg_print("............................................................")

  #
  ## If main called us - then return sysnset along with the pos_df
  if(my_parents_name == 'main'):
    dbg_print(pos_df)
    return (synsets, pos_df)
  else:
    return (synsets)



#
## ANCILIRY: load_data()
##-----------------------------------------------------------------------------
# load_data: this function will load the CSV into a pandas data-frame
# `docs_from_csv` is a DataFrame which contains the following columns: 
#
#   >> `Quality`
#   >> `D1`
#   >> `D2`
#
# Quality is an indicator variable which indicates if the two documents D1 and 
# D2 are docs_from_csv of one another (1 for paraphrase, 0 for not paraphrase).
#
# Use this dataframe for questions:
#
#  most_similar_docs 
#  label_accuracy
#
#------------------------------------------------------------------------------
#
#
# ld - method acronym
# 
def load_data(file_path):

  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  ''''
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_grand_parents_name} / {my_parents_name} - ')
  dbg_print("+------------------------------------------------------------")
  # dbg_print(f'  \_(ld) File to read: {file_path} ')
  '''
  
  docs_from_csv_file = None
  
  if(os.path.isfile(file_path)):
    try:
      docs_from_csv_file = pd.read_csv(file_path)
    except ValueError:
      dbg_print(f"Failed to read from file: {file_path}")
  else:
    dbg_print(f"File could not be found: {file_path}")

  return(docs_from_csv_file)  
  
  

#
## 
##-----------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------
#
#
# 
# 
def main():
  
  my_name, my_parents_name, my_grand_parents_name = stack_inspector()  
  
  # dbg_print(" ")
  dbg_print("\n")
  dbg_print("+------------------------------------------------------------")
  dbg_print(f'|   - {my_parents_name} / {my_name} - ')
  dbg_print("+------------------------------------------------------------")
  
  
  #
  ## Load pass-phrases from file
  data_file = 'data/paraphrases.csv'
  
  #
  ## Create a stop words list that will get passed around
  stop_words = set(stopwords.words('english'))
  salutation_stop_words = set(['mr', 'ms', 'mrs'])
  stop_words = stop_words | salutation_stop_words
  
     
  ''' 
  dbg_print(f' |.................................................................')
  dbg_print(f' |  >> English language stop words to use: {stop_words}')
  dbg_print(f' |.................................................................')
  ''' 

  #
  ## Call load_data to read the file content
  docs_from_csv = load_data(data_file)
  
  
  #
  ## Iterate over each row of the data-frame and call document_path_similarity
  ## (doc1, doc2)
  for index, row in docs_from_csv.iterrows():
    
    dbg_print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

    # dbg_print(row[["D1", "D2"]])
    paraphrase_value = row["Quality"]
    doc1 = row["D1"]
    doc2 = row["D2"]

    dbg_print(" +------------------------------------------------------------------")
    dbg_print(f' | \_(main) Document 1:')
    dbg_print(f' |.................................................................')
    dbg_print(f' |  >> {doc1}')
    dbg_print(f' |.................................................................')
    dbg_print(f' | \_(main) Document 2:')
    dbg_print(f' |.................................................................')
    dbg_print(f' |  >> {doc2}')
    dbg_print(f' |.................................................................')
    dbg_print(f' | \_(main) Docs are paraphrase:')
    dbg_print(f' |.................................................................')
    dbg_print(f' |  >> {paraphrase_value}')
    
    # doc1_synsets = doc_to_synsets(doc1, stop_words)
    # doc2_synsets = doc_to_synsets(doc2, stop_words)


    #
    ## Call doc_to_synsets to synsets and data-frame consisting of synsets for doc1
    (doc1_synsets, doc1_synset_df) = doc_to_synsets(doc1, stop_words)
    dbg_print(f'\n\nDoc-1 synset data-frame:\n {doc1_synset_df}')


    #
    ## Call doc_to_synsets to synsets and data-frame consisting of synsets for doc2    
    (doc2_synsets, doc2_synset_df) = doc_to_synsets(doc2, stop_words)    
    dbg_print(f'\n\nDoc-2 synset data-frame:\n {doc2_synset_df}')
    
    #
    ## Merge the two data-frames as two columnns into a new data-frame
    # Set the maximum column width to 20 characters
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.expand_frame_repr', False)
    synset_side_by_side_df = pd.concat([doc1_synset_df, doc2_synset_df], axis=1)
    
    dbg_print(f'\n\nMerged data-frame(s):\n {synset_side_by_side_df}')
    dbg_print(synset_side_by_side_df)
    
    
    #
    ## Pandas data-frame column headings are not wide enough, resize and convert to
    # table - this makes it possible to print the columns side-by-side
    # Modify the column names by truncating to a specified maximum width
    max_width = 500
    synset_side_by_side_df.columns = [name[:max_width] if len(name) > max_width else name for name in synset_side_by_side_df.columns]
    
    #
    # Convert DataFrame to tabular format
    merged_synset_table = tabulate(synset_side_by_side_df, headers='keys', tablefmt='grid')
    print('\n\n')
    print(f'|-----------------------------------------------------------------')
    print(f'| \_(main) Document 1 & 2 synsets side-by-side: ')
    print(f'|-----------------------------------------------------------------\n')
    print(merged_synset_table)
    

    dbg_print(f' |-----------------------------------------------------------------')
    dbg_print(f' | \_(main) Document 1 synsets is of type: {type(doc1_synsets)}')
    dbg_print(f' | \_(main) Document 1 synsets: {doc1_synsets}')
    dbg_print(f' |.................................................................')
    dbg_print(f' | \_(main) Document 1 synsets is of type: {type(doc2_synsets)}')
    dbg_print(f' | \_(main) Document 2 synsets: {doc2_synsets}')
    

    # dbg_print(tabulate(doc1_synsets))

    # dbg_print(row[["D1", "D2"]])
    
    #
    ## Calculate similarity score
    sim_score = similarity_score(doc1_synsets, doc2_synsets)
    print(f'| (main) Similarity score calculated by similarity_score: {sim_score} ')
    
    dbg_print(f' | \_(main)  Similarity score between:')
    dbg_print(f' | \_(main)   {doc1} and')
    dbg_print(f' | \_(main)   {doc2} is:')
    dbg_print(f' | \_(main)  {sim_score} ')

    dbg_print(f' |.................................................................')
    
    #
    ## Calculate document path similarity score
    doc_path_sim_score = document_path_similarity(doc1, doc2, stop_words)
    print(f'| (main) Path Similarity score from document_path_similarity : {doc_path_sim_score} ')
    
    #
    ## Print the paraphrase_value read in from the CSV
    print(f'| (main) Paraphrase value: {paraphrase_value} ')
    print('+-------------------------------------------------------------------')


    #
    ## Calculate document path similarity score
    ''' 
    doc_path_sim_score = document_path_similarity(doc1, doc2, stop_words)
    dbg_print(f' | (main) Path Similarity score :')
    dbg_print(f' | \_(main)   {doc1} and')
    dbg_print(f' | \_(main)   {doc2} is:')
    dbg_print(f' | (main)  {doc_path_sim_score} ')
    ''' 

    
    #
    ## Call test_document_path_similarity to test document_path_similarity
    # tested_similarity_score = test_document_path_similarity(stop_words)
    dbg_print(f' |.................................................................')
    # dbg_print(f' | \_(main) tested similarity score: {tested_similarity_score}')

    dbg_print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
  
  # '''
  
  ''' 
  dbg_print(f'|   Data-frame head: ')
  head = docs_from_csv.head(4)
  dbg_print(head)
  dbg_print(head[["D1", "D2"]])
  '''

  test_document_path_similarity(stop_words)
  ''' 
  test_document_path_similarity()
  most_similar_docs()
  ''' 
  
  # dbg_print(f'  \_(ld) Function name table: {func_name_dict}')
  


#
## START EXECUTION
#-----------------------------------------------------------------------------
#
if __name__ == "__main__":
  
  func_name_dict = {}
  
  #
  ## To generate call-graph run:
  ''' 
  with PyCallGraph(output=GraphvizOutput()):
    main()
  ''' 
  
  main()
  # dbg_print(f' Function name acronym table: {func_name_dict}')
  

  
  

