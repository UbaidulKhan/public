
import collections
import nltk
import numpy as np
import pandas as pd
from pprint import pprint
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
#

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
  


#
## add_feature - question 7 requires adding the length of the document to
## the sparse-matrix(X_train_tfidf_vectorizerorized)
##-----------------------------------------------------------------------------
def add_feature(X, feature_to_add):
  """
  Returns sparse feature matrix with added feature.
  feature_to_add can also be a list of features.
  """
  ''' 
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m add_feature \033[0m")
  print("+------------------------------------------------------------")
  '''
  
  # print(f'|  Adding feature {feature_to_add} to {X}')
  # print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

  return (hstack([X, csr_matrix(feature_to_add).T], 'csr'))


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
  
  print(f"|  series_data data type is: {type(series_data)}")
  
  #
  ## Create a list for storing the length of the text in X_train
  series_doc_len = []
  
  #
  ## Create a list for storing the length of the number of digits per document
  series_digits = []

  #
  ## Create a list for storing the number of non-alphanumeric characters
  ## to count_vectorizer
  series_nonalphas = []
  
  # entry = 0
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
    print(f'|  {idx} Text in the series: {text} ')
    print(f'|  Text length: {text_length} ')
    print(f'|  Number of digits in text: {text_digits} ')
    # print(f'|  Non-alpha-nums is of type: {type(text_non_alphas)}')
    print(f'|  There are {text_non_alphas_count} of non-alpha-nums in text: {text_non_alphas}')
    print(f'|  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= ')
    '''

    series_doc_len.append(text_length)
    series_digits.append(text_digits)
    series_nonalphas.append(text_non_alphas_count)
  

  #
  ## Convert X_train_text_length & X_train_non_alpha_count_series to pandas 
  ## series
  series_doc_len_series = pd.Series(series_doc_len)
  series_digits_series = pd.Series(series_digits)
  series_nonalphas_series = pd.Series(series_nonalphas)
  
  return(series_doc_len, series_digits, series_nonalphas)  
#
## Question 11
##-----------------------------------------------------------------------------
##
def answer_eleven(spam_data_df):
  
    vectorizer = CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=[2,5])

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_train_transformed_with_length = add_feature(X_train_transformed, [X_train.str.len(),
                                                                        X_train.apply(lambda x: len(''.join([a for a in x if a.isdigit()]))),
                                                                        X_train.str.findall(r'(\W)').str.len()])

    X_test_transformed = vectorizer.transform(X_test)
    X_test_transformed_with_length = add_feature(X_test_transformed, [X_test.str.len(),
                                                                      X_test.apply(lambda x: len(''.join([a for a in x if a.isdigit()]))),
                                                                      X_test.str.findall(r'(\W)').str.len()])

    clf = LogisticRegression(C=100)

    clf.fit(X_train_transformed_with_length, y_train)

    y_predicted = clf.predict(X_test_transformed_with_length)

    auc = roc_auc_score(y_test, y_predicted)

    feature_names = np.array(vectorizer.get_feature_names_out() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef_index = clf.coef_[0].argsort()
    smallest = feature_names[sorted_coef_index[:10]]
    largest = feature_names[sorted_coef_index[:-11:-1]]

    return (auc, list(smallest), list(largest))
  


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
  answer_eleven(df)
  
    