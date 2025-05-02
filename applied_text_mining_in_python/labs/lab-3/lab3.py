
#
#
##  Copyright(c), 2023
##  Ubaidul Khan, ubaidul.khan@gmail.com
##
##  File-name: lab3.py, Lab 3
##
##  Meta: 
##   Applied Text Mining in Python, lab-4
##   University of Michigan, Coursera
##

import collections
import nltk
import numpy as np
import pandas as pd
import pprint
from tabulate import tabulate
from datetime import datetime

# from tabulate import tabulate

#
##-------------------------- Answer 1 --------------------------
from sklearn.model_selection import train_test_split

#
##-------------------------- Answer 2 --------------------------
from sklearn.feature_extraction.text import CountVectorizer

#
##-------------------------- Answer 3 --------------------------
from sklearn.metrics import roc_auc_score

#
##-------------------------- Answer 4 --------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

#
##-------------------------- Answer 5 --------------------------
from sklearn.naive_bayes import MultinomialNB

#
##-------------------------- Answer 6 --------------------------
from scipy.sparse import csr_matrix, hstack

#
##-------------------------- Answer 7 --------------------------
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, hstack

#
##-------------------------- Answer 8 --------------------------
from sklearn.linear_model import LogisticRegression

#
##-------------------------- Answer 11 --------------------------
import re
from varname import nameof


#


#
## Lab URL:
## https://www.coursera.org/learn/python-text-mining/ungradedLab/HGLEa/assignment-3/lab?path=%2Fnotebooks%2FAssignment%25203.ipynb
##
##
#
## Additional Resource/Solution:
##  https://github.com/umer7/Applied-Text-Mining-in-Python/blob/master/Assignment%2B3.ipynb
##

#
## Sources found on the web:
#
''' 
   https://github.com/raj-aditya007/Applied-Text-Mining-python-Coursera/blob/master/Assignment%2B4.py
'''
##
#
# References:
#   
#  Creating feature and coefficient in a table:
#
##   https://stackoverflow.com/questions/34649969/how-to-find-the-features-names-of-the-coefficients-using-scikit-linear-regressio

#*                                                                            *


#
## This is an auxiliary function used for printing rows of table data
##-----------------------------------------------------------------------------
## 
## 
##
def row_printer(table, prefix):
  for row in table:
    to_print = prefix + row
    print(to_print)
 

#
## Pre-requisite functions - load data / train data
##-----------------------------------------------------------------------------
## What percentage of the documents in spam_data are spam?
## This function should return a float, the percent value 
## (i.e.  ratio∗100ratio∗100 ).
#
def load_data(file_name='data/spam.csv'):
  #
  ## Read the reviews into a data-frame
  spam_data_df = pd.read_csv(file_name)

  # print("\033[1m ")
  print("+------------------------------------------------------------")
  print("|  \033[1m --- load_data --- \033[0m")
  print("+------------------------------------------------------------")
  
  # print("|  before filtering")
  # print(spam_data_df.head())
  # pprint(spam_data_df.head(), indent=3)
  # pprint(f'Spam head data type is: {type(spam_data_df.head())}, indent=3')

  
  #
  ## Here we are converting the spam_data_df column values from spam/ham to
  ## 1 or 0.  If spam_data_df contains target column with values
  ## spam, then these will become 1, otherwise 0

  spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)
  # print("|  After filtering")
  # print(spam_data_df.head(10))
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  return(spam_data_df)


#
## Split Training Data
##-----------------------------------------------------------------------------
## Split data - test, train
#
def split_training_data(spam_data):
  X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                      spam_data['target'],
                                                      test_size=0.3, 
                                                      random_state=0)

  print("\n")
  print("+------------------------------------------------------------")
  print("|  \033[1m --- split_training_data --- \033[0m")
  print("+------------------------------------------------------------")
  
  # print(f'|   X_test is: {X_test} ')
  # print(f'|   X_test type is: {type(X_test)}')

  ''' 
  print(f'|  \033[1m X_train type is: {type(X_train)} \033[0m')
  print(f'|  \033[1m X_train shape is: {X_train.shape} \033[0m')

  print(f'|  \033[1m X_test type is: {type(X_test)} \033[0m')
  print(f'|  \033[1m X_test shape is: {X_test.shape} \033[0m')

  print(f'|  \033[1m Training every 20: {X_train[::20]} \033[0m')
  print(f'|  \033[1m Training label every 20: {y_train[::20]} \033[0m')
  '''
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

  return(X_train, X_test, y_train, y_test)  


#
## Question 1
##-----------------------------------------------------------------------------
## What percentage of the documents in spam_data are spam?
##
## Return: This function should return a float, the percent value 
## (i.e.  ratio∗100ratio∗100 ).
#
def answer_one(df):
  #
  ## There are two columns in the CSV:
  ##  1) text
  ##  2) target
  ##
  print("\n")
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 1 \033[0m")
  print("+------------------------------------------------------------")
  
  number_of_spam = len(df.loc[df['target'] == 1])
  print(f'|  Number of spam: {number_of_spam}')
  total_entries = len(df)
  print(f'|  Total number of entries: {total_entries}')
  # print("\033[1m ")

  
  print(f'|  Percentage of spam: {(number_of_spam/total_entries)*100}')
  # print("\033[0m")
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")


#
## Question test_training
##-----------------------------------------------------------------------------
## Fit the training data X_train using a Count Vectorizer with default 
## parameters.
##
## What is the longest token in the vocabulary?
##
## Return: This function should return a string..
#
def test_trainer(mnb_clf_model, vect, y_test):
  #
  ## Test the model with test data
  my_test = ['no way I am giving them my money', 
             'send money to african prince',
             'please send a million dollar to help poor africans',
             'I don\'t know u and u don\'t know me. Send CHAT and money',
             'Send me your email address and I will send youa million dollars',
             'XXXMobileMovieClub: To use your credit, click the WAP',
            ]

  #
  ## Convert the list to data-frame
  my_test_df = pd.Series(my_test)  
  # print(f'|   my_test_df type is: {type(my_test_df)}')
  # print(f'|   my_test_df contains: {my_test_df}')


  # vect = CountVectorizer().fit(my_test_df)
  my_test_vectorized = vect.transform(my_test_df)
  
  my_test_predictions = mnb_clf_model.predict(my_test_vectorized)

  

  # print(f'  my_test_vectorized type is: \n{type(my_test_vectorized)}')
  # print("-------------------------------------------------")
  
  
  ''' 
  print(f'  my_test_vectorized description: \n{my_test_vectorized.describe()}')
  print("-------------------------------------------------")
  print(f'  my_test_vectorized is: \n{my_test_vectorized}')
  print("-------------------------------------------------")
  '''

  one_count = zero_count = 0
  one_count_max = zero_count_max = (len(my_test))/2

  y_test_sub = y_test[1493:1499]  
  #
  ## Following code was used to find rows with value 1:
  # for row in y_test.items(): 
  #   print(f'  row type is: {type(row)}')
  #   if(row[1] == 1):
  #     # print (f'  y_test row is: {row}')
  #     if(one_count < one_count_max):
  #       y_test_sub.append(row)
  #       one_count += 1

  #   if(row[1] == 0):
  #     # print (f'  y_test row is: {row}')
  #     if(zero_count < zero_count_max):
  #       y_test_sub.append(row)
  #       zero_count += 1
  
  #
  ## Calculate the AUC:
  auc = roc_auc_score(y_test_sub, my_test_predictions)
  # auc = -1
    
  print("\n")
  print("+------------------------------------------------------------")
  print("|  \033[1m test_trainer \033[0m")
  print("+------------------------------------------------------------")
  print(f'| \033[1m Predictions for my_test_df: {my_test_predictions}\033[0m')
  print("+------------------------------------------------------------")
  print(f'|  my_test_df type is: {type(my_test_df)}')
  print("+------------------------------------------------------------")
  print(f'|  my_test_df length is: {len(my_test)}')
  print("+------------------------------------------------------------")
  print(f'|  my_test_df contains: \n{my_test_df}')
  print("+------------------------------------------------------------")
  print(f'|  y_test_sub type is: {type(y_test_sub)}')
  print("+------------------------------------------------------------")
  print(f'|  y_test_sub contains: \n{y_test_sub}')  
  print("+------------------------------------------------------------")
  print(f'|  \033[1m AUC score is:  {auc} \033[0m')
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  

#*                                                                            *

#
## Question 2
##-----------------------------------------------------------------------------
## Fit the training data X_train using a Count Vectorizer with default 
## parameters.
##
## What is the longest token in the vocabulary?
##
## Return: This function should return a string..
#
def answer_two(X_train):
  vect = CountVectorizer().fit(X_train)
  # print(f' Vocabulary in the vector: {vect.get_feature_names_out()[::2000]}')
  

  previous_word = ""
  longest_word = ""
  previous_lenth = 0
  max_length = 0
  
  for word in vect.get_feature_names_out():
    #
    ## Display the current word
    ## print(f'  Current word: {word}')
    #
    ## Calculate the lenght of the current word
    this_length = len(word)
    
    #
    ## If length of this word is longer than the previously stored length
    ## then this length is the longest.
    if(this_length > max_length):
      longest_word = word
      max_length = this_length
    else:
      continue

  print("\n")
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 2 \033[0m")
  print("+------------------------------------------------------------")
  
  print(f'|  Longest word is: {longest_word} of length {max_length}')
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    
  #
  # Longest word is: com1win150ppmx3age16subscription of length 32
  #
  return(vect)


#
## Question 3
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Count Vectorizer with 
## default parameters.
##
## Next, fit a multinomial Naive Bayes classifier model with smoothing 
## alpha=0.1. Find the area under the curve (AUC) score using the transformed 
## test data.
##
## Return: this function should return the AUC score as a float.
#
## References:
#  https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
#  Search for: Confusion matrix

def answer_three(X_train, X_test, y_train, y_test):

#*                                                                            *
  
  #
  ## We need to convert X_train into a numeric representation that 
  ## scikit-learn can use. The bag-of-words approach is simple and commonly 
  ## used way to represent text for use in machine learning, which ignores
  ## structure and only counts how often each word occurs.
  ##
  ## CountVectorizer allows us to use the bag-of-words approach by converting 
  ## a collection of text documents into a matrix of token counts.
  ##
  ## A document-term matrix or term-document matrix is a mathematical matrix 
  ## that describes the frequency of terms that occur in a collection of 
  ## documents. This is a matrix where:
  ##  
  ##  - Each row represents one document
  ##  - Each column represents one term (word)
  ## 
  ##
  ## First, we instantiate the CountVectorizer and fit it to our training 
  ## data.
  
  #
  ## Fitting the CountVectorizer consists of the tokenization of the trained 
  ## data and building of the vocabulary.
  vect = CountVectorizer().fit(X_train)

  #
  ## Fitting the CountVectorizer tokenizes each document by finding all 
  ## sequences of characters of at least two letters or numbers separated 
  ## by word boundaries. Converts everything to lowercase and builds a 
  ## vocabulary using these tokens.
  ##
  ## Transform documents to document-term matrix.
  ##
  X_train_vectorized = vect.transform(X_train)


  
#*                                                                            *
  #
  ## Create a multinomial Naive Bayes classifier model
  mnb_clf_model = MultinomialNB(alpha = 1.0, class_prior = None, 
                  fit_prior = True)
  
  #
  # Train the model
  mnb_clf_model.fit(X_train_vectorized, y_train)

  #
  ## Predict X_test
  predictions = mnb_clf_model.predict(vect.transform(X_test))

  #
  ## Calculate the AUC:
  auc = roc_auc_score(y_test, predictions)
  
  print("\n")
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 3 \033[0m")
  print("+------------------------------------------------------------")
  print(f'| \033[1m Predictions:  {predictions}\033[0m')
  print("+------------------------------------------------------------")
  # print(f'|  X_test type is: {type(X_test)}')
  # print("+------------------------------------------------------------")
  # print(f'|  X_test length is: {len(X_test)}')
  # print("+------------------------------------------------------------")
  # print(f'|  X_test contains: \n{X_test.head()}')
  # print("+------------------------------------------------------------")
  # print(f'|  y_test type is: {type(y_test)}')
  # print("+------------------------------------------------------------")
  # print(f'|  y_test head contains: \n{y_test.head()}')
  # print("+------------------------------------------------------------")  
  print(f'|  \033[1m AUC score is:  {auc} \033[0m')
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  
  return(mnb_clf_model, vect)

#*                                                                            *



#
## Question 4
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Tfidf Vectorizer with 
## default parameters.
##
## What 20 features have the smallest tf-idf and what 20 have the largest 
## tf-idf?
##
## Put these features in a two series where each series is sorted by tf-idf 
## value and then alphabetically by feature name. The index of the series 
## should be the feature name, and the data should be the tf-idf.
## 
## The series of 20 features with smallest tf-idfs should be sorted smallest 
## tfidf first, the list of 20 features with largest tf-idfs should be sorted 
## largest first.
## 
## 
## This function should return a tuple of two series (smallest tf-idfs series, 
## largest tf-idfs series).
##
#
## References:
##
##  https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/
##   https://stackoverflow.com/questions/36800654/how-is-the-tfidfvectorizer-in-scikit-learn-supposed-to-work
##
##  http://www.ultravioletanalytics.com/blog/tf-idf-basics-with-pandas-scikit-learn
##     
##  
##  
##
def answer_four(X_train, X_test, y_train, y_test):


  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 4 \033[0m")
  print("+------------------------------------------------------------")
  
  tfidf_vectorizer = TfidfVectorizer(use_idf=True)
  print(f'| tfidf_vectorizer data type is: {type(tfidf_vectorizer)}')

  # X_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(X_train)
  
  X_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(X_train)
  print(f'| X_train_tfidf_vectorized data type is: {type(X_train_tfidf_vectorized)}')
  
  ''' 
  feature_names_array = np.array(tfidf_vectorizer.get_feature_names_out())

  # place tf-idf values in a pandas data frame 
  tfidf_dense = X_train_tfidf_vectorized.todense() 
  tfidf_dense_list = tfidf_dense.tolist()
  tfidf_array = X_train_tfidf_vectorized.toarray()
  '''
  
  printer_prefix = "|  "
  
  feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
  sorted_tfidf_index = X_train_tfidf_vectorized.max(0).toarray()[0].argsort()

  print(f'| \033[1m Data type of the tfidf_index {type(sorted_tfidf_index)}:\033[0m')
  print(f'| \033[1m Dimensions of the numpy array {sorted_tfidf_index.ndim}:\033[0m')
  print(f'| \033[1m Shape of the numpy array {sorted_tfidf_index.shape}:\033[0m')

  
  twenty_features_smallest_tfidf = feature_names[sorted_tfidf_index[:20]]
  twenty_features_highest_tfidf = feature_names[sorted_tfidf_index[:-21:-1]]

  print("+------------------------------------------------------------")
  print('| \033[1m 20 Feature with smallest TF-IDF values:\033[0m')
  print("+------------------------------------------------------------")
  row_printer(twenty_features_smallest_tfidf, printer_prefix)
  print("+------------------------------------------------------------")
  print('| \033[1m 20 Feature with highest TF-IDF values:\033[0m')
  print("+------------------------------------------------------------")
  row_printer(twenty_features_highest_tfidf, printer_prefix)
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  # df = pd.DataFrame(tfidf_dense_list, index=feature_names)

  
  # place tf-idf values in a pandas data frame 
  # df = pd.DataFrame(X_train_tfidf_vectorized.todense(),
  #                  index = tfidf_vectorizer.get_feature_names()) 
       
       #            index=tfidf_vectorizer.get_feature_names_out()) 

       # columns=["tfidf"]) df.sort_values(by=["tfidf"],
       # ascending=False)

    
  return(tfidf_vectorizer)





#
## Question 5
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Tfidf Vectorizer ignoring 
## terms that have a document frequency strictly lower than 3.
## 
## Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1 
## and compute the area under the curve (AUC) score using the transformed test 
## data.
## 
## Return: This function should return the AUC score as a float.
##
def answer_five(X_train, X_test, y_train, y_test):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 5 \033[0m")
  print("+------------------------------------------------------------")
  
  #
  ## Create a TFIDF vectorizer, then fit the training data 
  tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=3)
  X_train_tfidf_vectorizerorized = tfidf_vectorizer.fit_transform(X_train)
  
  #
  ## Create a multinomial Naive Bayes classifier model
  mnb_clf_model = MultinomialNB(alpha = 0.1, class_prior = None, 
                  fit_prior = True)
  
  #
  # Train the model
  mnb_clf_model.fit(X_train_tfidf_vectorizerorized, y_train)

  #
  ## Predict X_test
  predictions = mnb_clf_model.predict(tfidf_vectorizer.transform(X_test))
  # print(f'   Predictions is of type: {print(type(predictions))}')

  #
  ## Calculate the AUC:
  auc = roc_auc_score(y_test, predictions)

  print(f'|  \033[1m AUC score is:  {auc} \033[0m')
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  #
  ## Return the auc score
  return(auc, predictions)


#
## Question 6 - speedy, this utilizes the pandas data-frame functionality
##-----------------------------------------------------------------------------
## What is the average length of documents (number of characters) for not spam 
## and spam documents?
##
## Return: This function should return a tuple (average length not spam, average 
## length spam).
##
##
## Resource:
##
## https://stackoverflow.com/questions/42815768/pandas-adding-column-with-the-length-of-other-column-as-value
##
##
def answer_six(spam_data_df):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 6 \033[0m")
  print("+------------------------------------------------------------")
    
  spam_data_df['text_length'] = spam_data_df['text'].str.len()
  
  ham_length = np.mean(spam_data_df['text_length'][spam_data_df['target'] == 0])
  
  spam_length = np.mean(spam_data_df['text_length'][spam_data_df['target'] == 1])

  print(f"|  Ham text averag lenth: {ham_length}")
  print(f"|  Spam text averag lenth: {spam_length}")
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

  return(ham_length, spam_length)
  
  

#
## Question 6 - lengthy version
##-----------------------------------------------------------------------------
## What is the average length of documents (number of characters) for not spam 
## and spam documents?
##
## Return: This function should return a tuple (average length not spam, average 
## length spam).
##
## https://github.com/umer7/Applied-Text-Mining-in-Python/blob/master/Assignment%2B3.ipynb
##
## Pandas Vectorization problems/solutions:
##
##  https://pythonspeed.com/articles/pandas-vectorization/
##  https://stackoverflow.com/questions/24870953/does-pandas-iterrows-have-performance-issues
##
##
def answer_six_uk(spam_data_df):

  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 6 - lengthy\033[0m")
  print("+------------------------------------------------------------")
  
  
  #
  ## Print column names
  '''     
  for col in spam_data_df.columns:
      print(f'   Column name is: {col}')
  '''

  #
  ## Separate the spam/ham data-frames 
  spam_array = spam_data_df.loc[spam_data_df['target'] == 1]
  ham_array = spam_data_df.loc[spam_data_df['target'] == 0]

  # print(spam_array)
  # print(type(spam_array))

  #
  ## Iterate over the spam data-frame and pull out the text field - this is
  ## the first field. Second field is the target. There are TWO fieldsare: 
  ##  text 
  ##  target 
  #
  ## These two variables are used for calculating the avarage length
  ## of ham documents  
  ham_doc_length_total = 0 
  ham_doc_total_count = 0 
  #
  for i in range(len(ham_array)):
    # print(f'  {i} entry in ham is: {ham_array.iloc[i, 0]}\n')
    this_ham_doc_length = len(ham_array.iloc[i, 0])
    ham_doc_length_total += this_ham_doc_length
    ham_doc_total_count = i
    # print(f'{i} entry in ham is: {ham_array.iloc[i, 0]}\n')
    
    ''' 
    print(f'  {i} entry in ham has length: {this_ham_doc_length}\n')
    print(f'  {i} ham doc total length: {ham_doc_length_total}\n')
    print(f'  {i} ham doc total count: {ham_doc_total_count}\n')
    ''' 


  print(f'|  ham doc total length: {ham_doc_length_total}')
  print(f'|  ham doc total count: {ham_doc_total_count}')

  #
  ## Calculate the lenght of spam documents 
  ##   
  #
  ## These two variables are used for calculating the avarage length
  ## of spam documents
  spam_doc_length_total = 0 
  spam_doc_total_count = 0 
  #
  for i in range(len(spam_array)):
    # print(f'  {i} entry in spam is: {spam_array.iloc[i, 0]}\n')
    this_spam_doc_length = len(spam_array.iloc[i, 0])
    spam_doc_length_total += this_spam_doc_length
    spam_doc_total_count = i

    ''' 
    # print(f'{i} entry in spam is: {spam_array.iloc[i, 0]}\n')
    print(f'  {i} entry in spam has length: {this_spam_doc_length}\n')
    print(f'  {i} spam doc total length: {spam_doc_length_total}\n')
    print(f'  {i} spam total count: {spam_doc_total_count}\n')
    '''

  print(f'|  spam doc total length: {spam_doc_length_total}')
  print(f'|  spam doc total count: {spam_doc_total_count}')
      
  ham_avg_length = (ham_doc_length_total/ham_doc_total_count)
  spam_avg_length = (spam_doc_length_total/spam_doc_total_count)

  print(f'| \033[1m Ham average length is:  {ham_avg_length} \033[0m')
  print(f'| \033[1m Spam average length is:  {spam_avg_length} \033[0m')

  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  return(ham_avg_length, spam_avg_length)


#
## add_feature - question 7 requires adding the length of the document to
## the sparse-matrix(X_train_tfidf_vectorizerorized)
##-----------------------------------------------------------------------------
def add_feature(X, feature_to_add):
  """
  Returns sparse feature matrix with added feature.
  feature_to_add can also be a list of features.
  """
  
  #''' 
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m --- add_feature --- \033[0m")
  print("+------------------------------------------------------------")
  #'''
  # print(f'|  Adding feature {feature_to_add} to {X}')
  print(f'|  Adding feature feature_to_add to X')
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  print(f'|  \033[1m nameof(X) is of type: {type(X)} \033[0m')
  print(f'|  \033[1m X shape(rows/columns) before modification is: {X.shape} \033[0m')
  X_modified = hstack([X, csr_matrix(feature_to_add).T], 'csr')
  print(f'|  \033[1m X_modified shape(rows/columns) after modification is: {X_modified.shape} \033[0m')

  
  return(X_modified)


#
## Question 7
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Tfidf Vectorizer 
## ignoring terms that have a document frequency strictly lower than 5.
## 
## Using this document-term matrix add an additional feature, the length of 
## document (number of characters), fit a Support Vector Classification model 
## with regularization C=10000. Then compute the area under the curve (AUC) 
## score using the transformed test data.
##
##
## Return:This function should return the AUC score as a float.
##
## Reference(s):
##  hstack
##
##  https://scipython.com/book/chapter-6-numpy/examples/vstack-and-hstack/
##  https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.sparse.hstack.html

# def answer_seven(X_train, X_test, y_train, y_test, ham_avg_length, spam_avg_length):
def answer_seven():
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 7 \033[0m")
  print("+------------------------------------------------------------")
  
  #
  ## Create a Tfidf Vectorizer ignoring terms that have a document frequency 
  ## strictly lower than 5. 
  tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=5)

  #
  ## Vectorize X_train and X_train with length
  X_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(X_train)
  X_train_tfidf_vectorized_with_length = add_feature(X_train_tfidf_vectorized, X_train.str.len())

  
  #
  ## Vectorize X_test and then add document length to X_test 
  X_test_transformed_tfidf_vectorized = tfidf_vectorizer.transform(X_test)
  X_test_transformed_tfidf_vectorized_with_length = add_feature(X_test_transformed_tfidf_vectorized, X_test.str.len())

  
  #
  ## Create a classifier model 
  classifier_svc = SVC(C=10000)
  
  #
  ## Fit the training data 
  classifier_svc.fit(X_train_tfidf_vectorized_with_length, y_train)
  
  #
  ## Predict the X_test_tfidf_vectorizer
  y_predicted = classifier_svc.predict(X_test_transformed_tfidf_vectorized_with_length)
  

  #
  ## calculate the AUC score 
  auc_score = roc_auc_score(y_test, y_predicted)
  
  print(f'| \033[1m AUC Score is:  {auc_score} \033[0m')

  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  return(auc_score)

    

#
## Question 8
##-----------------------------------------------------------------------------
## What is the average number of digits per document for not spam and spam 
## documents?
##
## Return: This function should return a tuple (average # digits not spam, 
##         average # digits spam).
##
def answer_eight(spam_data_df):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 8 \033[0m")
  print("+------------------------------------------------------------")
    
  spam_data_df['num_digits'] = spam_data_df['text'].str.count('\d')
  
  ham_avg_digits = np.mean(spam_data_df['num_digits'][spam_data_df['target'] == 0])
  
  spam_avg_digits = np.mean(spam_data_df['num_digits'][spam_data_df['target'] == 1])

  print(f"|  Ham average digits: {ham_avg_digits}")
  print(f"|  Spam average digits: {spam_avg_digits}")
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

  return(ham_avg_digits, spam_avg_digits)
  


#
## Question 9
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Tfidf Vectorizer 
## ignoring terms that have a document frequency strictly lower than 5 and
## using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams).
## 
## Using this document-term matrix and the following additional features:
## 
## the length of document (number of characters)
## number of digits per document
## fit a Logistic Regression model with regularization C=100. Then compute the 
## area under the curve (AUC) score using the transformed test data.
##
##
## Return: This function should return the AUC score as a float.
##
def answer_nine(df, avg_len_ham, avg_len_spam, ham_avg_digits, spam_avg_digits):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 9 \033[0m")
  print("+------------------------------------------------------------")
    
  #
  ## Create a Tfidf Vectorizer ignoring terms that have a document frequency 
  ## strictly lower than 5 and using word n-grams from n=1 to n=3 
  tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=5, ngram_range=(1,3))
  #return()
  
  #
  ## df consists of two columns:
  ##  - text 
  ##  - target 
  
  #
  ## Vectorize X_train
  X_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(X_train)
  
  #
  ## Append the sparse matrix by adding text-length
  X_train_tfidf_vectorized_with_text_length = add_feature(X_train_tfidf_vectorized, X_train.str.len())
  
  #
  ## Append the sparse matrix by adding number-of-digits in the text
  X_train_tfidf_vectorized_with_text_length_digts = add_feature(X_train_tfidf_vectorized_with_text_length, X_train.str.count('\d'))

  
  #
  ## Vectorize X_test and then add document length to X_test 
  X_test_transformed_tfidf_vectorized = tfidf_vectorizer.transform(X_test)
  
  #
  ## Append the sparse matrix by adding text-length
  X_test_transformed_tfidf_vectorized_with_length = add_feature(X_test_transformed_tfidf_vectorized, X_test.str.len())
  
  #
  ## Append the sparse matrix by adding number-of-digits in the text
  X_test_tfidf_vectorized_with_text_length_digts = add_feature(X_test_transformed_tfidf_vectorized_with_length, X_test.str.count('\d'))

  
  #
  ## Create a classifier model 
  classifier_svc = LogisticRegression(C=10000, solver='liblinear')
  
  #
  ## Fit the training data 
  classifier_svc.fit(X_train_tfidf_vectorized_with_text_length_digts, y_train)
  
  #
  ## Predict the X_test_tfidf_vectorizer
  y_predicted = classifier_svc.predict(X_test_tfidf_vectorized_with_text_length_digts)
  

  #
  ## calculate the AUC score 
  auc_score = roc_auc_score(y_test, y_predicted)
  
  print(f"|  AUC score: {auc_score}")
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")



#
## Question 10
##-----------------------------------------------------------------------------
## What is the average number of non-word characters (anything other than a 
## letter, digit or underscore) per document for not spam and spam documents?
## 
## Hint: Use \w and \W character classes
##
## Return: This function should return a tuple (average # non-word characters 
## not spam, average # non-word characters spam).
##
## References:
##  https://pynative.com/python-regex-special-sequences-and-character-classes/#h-special-sequence-w-and-w
##
def answer_ten(spam_data_df):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 10 \033[0m")
  print("+------------------------------------------------------------")
  
  # search_string = "a-zA-Z0-9"
    
  spam_data_df['non_alphanum'] = spam_data_df['text'].str.findall(r'(\W)').str.len()
  
  ham_non_alphanum = np.mean(spam_data_df['non_alphanum'][spam_data_df['target'] == 0])
  spam_non_alphanum = np.mean(spam_data_df['non_alphanum'][spam_data_df['target'] == 1])

  print(f"|  Average non-alphanumeric in spam: {spam_non_alphanum}")
  print(f"|  Average non-alphanumeric in ham: {ham_non_alphanum}")

  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  
  return(ham_non_alphanum, spam_non_alphanum)



#
## This function will update the data-frame, but adding following:
## columns:
##
##  * Length of text
##  * Number of digits
##  * Number of non-alpha-numerics
##
def update_dtm_add_columns(series_data):
  pass



#
## This function will update the data-frame, but adding following:
## columns:
##
##  * Length of text
##  * Number of digits
##  * Number of non-alpha-numerics
##
def update_x_data(series_data):
  
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m --- update_x_data --- \033[0m")
  print("+------------------------------------------------------------")  
  
  print(f"|  update_x_data:: date-type of series_data is: {type(series_data)}")
  print(f"|  update_x_data:: Columans in {nameof(series_data)}: {series_data.name}")

  
  #
  ## Create a list for storing the length of the text in X_train
  series_doc_len = []
  
  #
  ## Create a list for storing the length of the number of digits per document
  series_digits = []

  #
  ## Create a list for storing the number of non-alphanumeric characters
  ## to count_vectorizer
  series_non_alphas = []
  
  #
  ## Convert series to data-frame
  data_frame = series_data.to_frame()
  
  #
  ## Series to return
  series_to_return = None
  
  #
  ## Get column names from data-frame
  for columns in data_frame.columns:
    print(f"|  Column names are: {columns}")
  
  #
  ## Iterate over the data-frame nad calculate length 
  ## and non-alphas
  for index, row in data_frame.iterrows():
    
    #
    ## Pull the text message 
    text_message = row['text']

    #
    ## Calculate and store the length of the text_message
    text_message_length = (len(text_message))
    series_doc_len.append(text_message_length)
    # data_frame['length_of_doc'] = text_message_length
    # print(f"|  Text message length: {text_message_length}")

    #
    ## Calculate and store the number of digits    
    text_message_digits = sum(c.isdigit() for c in text_message)
    series_digits.append(text_message_digits)
    # data_frame['digit_count'] = text_message_digits
    # print(f"|  Digits in text message: {text_message_digits}")

    
    #
    ## non_alpha =  re.search(r'\W+', text_message)
    text_message_non_alphas =  re.findall(r'\W+', text_message)
    text_message_non_alphas_count =  len(text_message_non_alphas)
    series_non_alphas.append(text_message_non_alphas_count)
    # data_frame['non_word_char_count'] = text_message_non_alphas_count
    # print(f"|  Non-alphas in text message: {text_message_non_alphas_count}")
    

  #
  ## Update the data-frame - append with columns  
  data_frame['length_of_doc'] = series_doc_len
  data_frame['digit_count'] = series_digits
  data_frame['non_word_char_count'] = series_non_alphas

  #
  ## Convert data-frame to numpy series and return the series

#
## Feature Extractor - this function will recieve a pandas series and inspect
## for certain features. Then it will sum/tally up the occurances of these
## features:
##
##  * Length of text
##  * Number of digits
##  * Number of non-alpha-numerics
##
##-----------------------------------------------------------------------------
## 
def feature_extractor(series_data):

  # print("\033[1m ")
  print('\n')
  print("+------------------------------------------------------------")
  print("|  \033[1m --- feature_extractor --- \033[0m")
  print("+------------------------------------------------------------")
  # pp = pprint.PrettyPrinter(indent=5)

  print(f"|  >> feature_extractor:: series_data data type is: {type(series_data)}")
  
  #
  ## Create a list for storing the length of the text in X_train
  series_doc_len = []
  
  #
  ## Create a list for storing the length of the number of digits per document
  series_digits = []

  #
  ## Create a list for storing the number of non-alphanumeric characters
  ## to count_vectorizer
  series_non_alphas = []
  
  entry = 0
  for (idx, text) in enumerate(series_data):
    
    # Calculate and store the length of the text
    text_length = (len(text))

    # Calculate and store the number of digits    
    # text_digit_count = text.count('\d')
    text_digits = sum(c.isdigit() for c in text)
    
    # non_alpha =  re.search(r'\W+', text)
    text_non_alphas =  re.findall(r'\W+', text)
    text_non_alphas_count =  len(text_non_alphas)
    
    ''' 
    print(f'|  feature_extractor:: Processing {idx}::{entry} of the series')
    print(f'|  feature_extractor:: Text: {text}')
    print(f'|  feature_extractor:: Text length: {text_length}')
    print(f'|  feature_extractor:: Number of digits in text: {text_digits}')
    # print(f'|  feature_extractor:: Non-alpha-nums is of type: {type(text_non_alphas)}')
    print(f'|  feature_extractor:: There are {text_non_alphas_count} of non-alpha-nums in text: {text_non_alphas}')
    print(f'|  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ')
    '''

    series_doc_len.append(text_length)
    series_digits.append(text_digits)
    series_non_alphas.append(text_non_alphas_count)
    
    #
    ## Increment the counter
    entry += 1
  

  #
  ## Convert X_train_text_length & X_train_non_alpha_count_series to pandas 
  ## series
  series_doc_len_series = pd.Series(series_doc_len)
  series_digits_series = pd.Series(series_digits)
  series_non_alphas_series = pd.Series(series_non_alphas)
  
  #
  ## Rename the series
  series_doc_len_renamed = series_doc_len_series.rename('length_of_doc')
  series_digits_renamed =  series_digits_series.rename('digit_count')
  series_non_alphas_renamed = series_non_alphas_series.rename('non_word_char_count')
  
  
  header = "Document Length"
  
  # Generate the table in fancy format.
  print(f'|  feature_extractor:: type of series_doc_len is: {type(series_doc_len)}')
  # table = tabulate(series_doc_len, header, tablefmt="fancy_grid")
  # print(table)
  
  ''' 
  print("   >> feature_extractor::\n")

  print(f'   >> feature_extractor::\n {series_digits_renamed}')
  print(f'   >> feature_extractor::\n {series_non_alphas_renamed}')
  '''
  
  return(series_doc_len_renamed, series_digits_renamed, series_non_alphas_renamed)  


#
## This data-frame will convert a sparse-matrix to data-frame and then
## print the data-frame
##
def print_sparse_matrix(matrix):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m --- print_sparse_matrix --- \033[0m")
  print("+------------------------------------------------------------")
  
  my_data_frame = pd.DataFrame.sparse.from_spmatrix(matrix) 

  for column_names in my_data_frame.columns:
    print(f'|  print_sparse_matrix:: column name is: {column_names}')
    
  print(my_data_frame.head()) 


  
#
## Question 11
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Count Vectorizer 
## ignoring terms that have a document frequency strictly lower than 5 and 
## using character n-grams from n=2 to n=5.
## 
## To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' 
## which creates character n-grams only from text inside word boundaries. This 
## should make the model more robust to spelling mistakes.
## 
## Using this document-term matrix and the following additional features:
## 
##  * The length of document (number of characters)
##  * Number of digits per document
##  * Number of non-word characters (anything other than a letter, digit or 
##    underscore.)
##
##  fit a Logistic Regression model with regularization C=100. Then compute the 
##  area under the curve (AUC) score using the transformed test data.
##  
##  Also find the 10 smallest and 10 largest coefficients from the model and return 
##  them along with the AUC score in a tuple.
##  
##  The list of 10 smallest coefficients should be sorted smallest first, the list 
##  of 10 largest coefficients should be sorted largest first.
##  
##  The three features that were added to the document term matrix should have 
##  the following names should they appear in the list of 
##  
##  coefficients: ['length_of_doc', 'digit_count', 'non_word_char_count']
##  
##  This function should return a tuple (AUC score as a float, smallest coefs list, 
##  largest coefs list).
##
##
## References:
##
##  Interpreting coefficients of linear models
##  https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
#
## Logistic Regression
##  https://www.youtube.com/watch?v=vN5cNN2-HWE
##
## Logistic Regression - Log Odds
##
##  https://www.youtube.com/watch?v=ARfXDSkQf1Y
##
## Odds Ratio / Log odds
##
##  https://www.youtube.com/watch?v=8nm0G-1uJzA

##
def answer_eleven():
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 11 \033[0m")
  print("+------------------------------------------------------------")

  #
  ## df consists of two columns:
  ##  - text 
  ##  - target 
  

  print("| \033[1m Vectorizing Training Data \033[0m")

    
  #
  ## Create a Tfidf Vectorizer ignoring terms that have a document frequency 
  ## strictly lower than 5. 
  vectorizer = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb')

  #
  ## Learn the vocabulary dictionary and return document-term matrix.
  X_train_vectorized = vectorizer.fit_transform(X_train)
  
  #
  ## Display the CountVectorizer vocabulary & length:
  # print(f"|  Vectorizer has : {len(vectorizer.vocabulary_)} tokens")
  # print(f"|  Vocabulary in Vectorizing: {vectorizer.vocabulary_}")

  
  #
  ## summarize encoded vector
  print(f"| \033[1m Vectorizer shape(rows/columns): {X_train_vectorized.shape} tokens \033[0m ")
  # print(vector.toarray())
  
  # print(f"|  Vectorizing shape after fit_transform: {vectorizer.shape}")

  
  #
  ## From X_train(pandas data series) get additional features
  (X_train_doclen, X_train_numdigits, X_train_nonalpha) = feature_extractor(X_train) 
  
  #
  ## Iterate over the data returned by feature_extractor and add each
  ## individual feature series to X_train_vectorized
  for feature in (X_train_doclen, X_train_numdigits, X_train_nonalpha):
    print(f"|  nameof(feature) is of type: {type(feature)}")
    # print(f"|  Appending feature {nameof(feature)} to X_train_vectorized")
    X_train_vectorized = add_feature(X_train_vectorized, feature)
    # print(f"|  nameof(X_train_vectorized) is of type: {type(X_train_vectorized)}")

    
  print("| \033[1m Vectorizing Testing Data \033[0m")

  #
  ## Create a sparse-matrix from X_test 
  X_test_vectorized = vectorizer.transform(X_test)
  
  ## From X_train(pandas data series) get additional features
  (X_test_doclen, X_test_numdigits, X_test_nonalpha) = feature_extractor(X_test) 
  
  # X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
  # X_test_vectorized = add_feature(X_test_vectorized, X_test_doclen)
  # X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())

  #
  ## Iterate over the data returned by feature_extractor and add each
  ## individual feature series to X_test_vectorized
  run_count=0
  for feature in (X_test_doclen, X_test_numdigits, X_test_nonalpha):
    # print(f"|  Appending feature {nameof(feature)} to X_test_vectorized")
    # print(f"| \033[1m Vectorizer after reshaping {run_count} \033[0m ")
    # print(f"| \033[1mshape(rows/columns): {X_train_vectorized.shape} tokens \033[0m ")
    X_test_vectorized = add_feature(X_test_vectorized, feature)
    run_count += 1
    
  ''' 
  print(f"|  Document Term Matix is of type: {type(X_test_vectorized)}")
  print(f"|  Document Term Matix has the shape(Rows x Column): {X_test_vectorized.shape}")
  ''' 
  
  
  #
  ## Create a logistic-regression 
  # classifier = SVC(C=10000)
  classifier = LogisticRegression(C=100, solver='liblinear')
  
  #
  ## Fit the training data 
  classifier.fit(X_train_vectorized, y_train)
  
  #
  ## Predict the X_test_vectorizer
  y_predicted = classifier.predict(X_test_vectorized)
  
  #
  ## Calculate the AUC score
  auc = roc_auc_score(y_test, y_predicted)


  #
  ## Pull all the coefficients
  all_coefficients = classifier.coef_
  print(f"|  All coefficients is of type: {type(all_coefficients)}")
  
  #
  ## Pull coefficients and sort them
  # sorted_coef_index = classifier.coef_[0].argsort()
  # print(f"|  Sorted coefficients_0 are of type: {type(sorted_coef_index)}")
  # print(f"|  Sorted coefficients_0 is of shape: {sorted_coef_index.shape}")
    
  # smallest_coefficients = sorted_coef_index[:10]
  # largest_coefficients = sorted_coef_index[-10:]
  
  # print(f"|  Ten smallest coefficients: {smallest_coefficients}")
  # print(f"|  Ten largest coefficients: {largest_coefficients}")


  ''' 
  sorted_index = X_train_vectorized.max(0).toarray()[0].argsort()
  '''
  
  # feature_names = np.array(vectorizer.get_feature_names_out())
  feature_names = vectorizer.get_feature_names_out()
  feature_names_array = np.array(feature_names)
  
  ''' 
  print(f"|  *** feature_names is of type: {type(feature_names)}")
  print(f"|  *** feature_names_array has the shape: {feature_names_array.shape}")

  print(f"|  *** feature_names_array is of type: {type(feature_names_array)}")
  print(f"|  feature_names_array: {feature_names_array}")
  '''
  
  #
  ## push new feature names into feature_names array
  feature_names_array = np.append(feature_names_array, 'length_of_doc')
  feature_names_array = np.append(feature_names_array, 'digit_count')
  feature_names_array = np.append(feature_names_array, 'non_word_char_count')
  print(f"|  *** After addition - feature_names_array has the shape: {feature_names_array.shape}")


  coef_rankings = pd.DataFrame(
    classifier.coef_[0].argsort(),
    columns = ['Coefficient'],
    index = feature_names_array,
  )

  #
  ## Rename index column - initially there is no column header for the index
  coef_rankings = coef_rankings.rename_axis('Feature')

  
  #
  ## Sort the data-frame by coefficient column
  coef_rankings = coef_rankings.sort_values(by=['Coefficient'])
  
  #
  ## Print the data-frame in a SQL like tabular format.
  # print(tabulate(coef_rankings.sort_values(by=['Coefficient']), headers='keys', tablefmt='psql'))
  

  smallest_coefficients_with_name = coef_rankings.head(n=10)
  print(f"\n  Smallest coefficients:")
  print(tabulate(smallest_coefficients_with_name.sort_values(by=['Coefficient']), headers='keys', tablefmt='psql'))
  
  largest_coefficients_with_name = coef_rankings.tail(n=10)
  largest_coefficients_sorted = largest_coefficients_with_name.sort_values(by=['Coefficient'], ascending=False)
  
  print(f"\n  Largest coefficients:")
  print(tabulate(largest_coefficients_sorted, headers='keys', tablefmt='psql'))
  

  return(auc, smallest_coefficients_with_name, largest_coefficients_sorted)
  
  
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=
#                         MAIN Starts below....
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=

if __name__ == "__main__":
  df = load_data()
  #
  ## There are two columns in the CSV:
  ##  1) Text
  ##  2) Target
  ##
  # print(f'  Length of data frame is: {len(df)}')
  # print(f'  Head of the data frame: {df.head()}')


  answer_one(df)
  
  #
  ## Split data into training and test
  X_train, X_test, y_train, y_test = split_training_data(df)
  
  #
  ## Print X_train and y_train:
  # print(f'  Xtrain is: {X_train}')
  
  #
  ## Question-2: Get count vectorizer from answer_two:
  vectorizer = answer_two(X_train)
  
  
  #
  ## Question-3: fit a multinomial Naive Bayes classifier model
  mnb_clf_model, vectorizer = answer_three(X_train, X_test, y_train, y_test)
  
  #
  ## Test the trainer with in-line test data
  # test_trainer(mnb_clf_model, vectorizer, y_test)
  
  #
  ## Question-4: fit a multinomial Naive Bayes classifier model
  # vectorizer = answer_four(X_train, X_test, y_train, y_test)
  # vectorizer = answer_four_take2(X_train, X_test, y_train, y_test)
  tfidf_vectorizer = answer_four(X_train, X_test, y_train, y_test)
  
  
  #
  ## Question-5: 
  ## Fit and transform the training data X_train using a Tfidf Vectorizer ignoring 
  ## terms that have a document frequency strictly lower than 3.
  auc_score, predictions = answer_five(X_train, X_test, y_train, y_test)
  
  
  #
  ## Question-6: 
  ## What is the average length of documents (number of characters) for not spam 
  ## and spam documents?
  ##
  ## Return: This function should return a tuple (average length not spam, average 
  ## length spam).
  # start = datetime.now()
  (avg_len_ham, avg_len_spam) = answer_six(df)
  #end = datetime.now()
  #duration = (end - start)
  #print(f'| Duration of answer_six short is: {duration}')

  # start = datetime.now()  
  (avg_len_ham, avg_len_spam) = answer_six_uk(df)
  # end = datetime.now()
  # duration = (end - start)
  # print(f'|  Duration of answer_six long is: {duration}')
  
    
  #
  ## Question-7: 
  ## Fit and transform the training data X_train using a Tfidf Vectorizer ignoring 
  ## terms that have a document frequency strictly lower than 5.
  #
  ## Using this document-term matrix and an additional feature, the length of document 
  ## (number of characters), fit a Support Vector Classification model with regularization 
  ## C=10000. Then compute the area under the curve (AUC) score using the transformed test data.
  #
  ## This function should return the AUC score as a float.
  auc_score = answer_seven()

  
  #
  ## Question-8: what is the average number of digits per document for not spam and 
  ## spam documents?
  ##
  ## This function should return a tuple (average # digits not spam, average # digits 
  ## spam).*
  (ham_avg_digits, spam_avg_digits) = answer_eight(df)
  
  
  #
  ## Question-9: 
  #
  ## Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that 
  ## have a document frequency strictly lower than 5 and using word n-grams from n=1 to n=3 
  ## (unigrams, bigrams, and trigrams).
  answer_nine(df, avg_len_ham, avg_len_spam, ham_avg_digits, spam_avg_digits)
  
  
  
  #
  ## Question 10
  #
  ## What is the average number of non-word characters (anything other than a letter, digit or 
  ## underscore) per document for not spam and spam documents?
  answer_ten(df)
  
  
  #
  ## Question 11
  ## Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have 
  ## a document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.
  #
  ## To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' which creates 
  ## character n-grams only from text inside word boundaries. This should make the model more 
  ## robust to spelling mistakes.
  #
  ## Using this document-term matrix and the following additional features:
  #
  ##  1) The length of document (number of characters)
  ##  2) Number of digits per document
  ##  3) Number of non-word characters (anything other than a letter, digit or underscore.)
  (auc, small_coefs, large_coeffs) = answer_eleven()
  
    