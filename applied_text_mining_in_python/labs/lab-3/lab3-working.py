
import collections
import nltk
import numpy as np
import pandas as pd
from pprint import pprint
#
##-------------------------- Answer 1 --------------------------
from sklearn.model_selection import train_test_split
#
##-------------------------- Answer 2 --------------------------
from sklearn.feature_extraction.text import CountVectorizer
#
##-------------------------- Answer 3 --------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
#
##-------------------------- Answer 4 --------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
#
##-------------------------- Answer 5 --------------------------
#
##-------------------------- Answer 6 --------------------------
from scipy.sparse import csr_matrix, hstack
#
##-------------------------- Answer 7 --------------------------
from sklearn.svm import SVC
#
##-------------------------- Answer 9 --------------------------
from sklearn.linear_model import LogisticRegression

#
## Lab URL:
## https://www.coursera.org/learn/python-text-mining/ungradedLab/HGLEa/assignment-3/lab?path=%2Fnotebooks%2FAssignment%25203.ipynb
##
##
#
## Additional Resource/Solution:
##  https://github.com/umer7/Applied-Text-Mining-in-Python/blob/master/Assignment%2B3.ipynb


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
  print("|  before filtering")
  print(spam_data_df.head())

  
  #
  ## Here we are converting the spam_data_df column values from spam/ham to
  ## 1 or 0.  If spam_data_df contains target column with values
  ## spam, then these will become 1, otherwise 0

  spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)
  print("|  After filtering")
  print(spam_data_df.head(10))
  
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
  ''' 
  print(f'|  \033[1m X_train type is: {type(X_train)} \033[0m')
  print(f'|  \033[1m X_train shape is: {X_train.shape} \033[0m')

  print(f'|  \033[1m X_test type is: {type(X_test)} \033[0m')
  print(f'|  \033[1m X_test shape is: {X_test.shape} \033[0m')

  print(f'|  \033[1m Training every 20: {X_train[::20]} \033[0m')
  print(f'|  \033[1m Training label every 20: {y_train[::20]} \033[0m')
  '''
    
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
  number_of_spam = len(df.loc[df['target'] == 'spam'])
  print(f' Number of spam: {number_of_spam}')
  total_entries = len(df)
  print(f' Total number of entries: {total_entries}')
  # print("\033[1m ")
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 1 \033[0m")
  print("+------------------------------------------------------------")
  
  print(f'|  Percentage of spam: {(number_of_spam/total_entries)*100}')
  # print("\033[0m")


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

  # print("\033[1m ")
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 2 \033[0m")
  print("+------------------------------------------------------------")
  
  print(f'|  Longest word is: {longest_word} of length {max_length}')
  # print("\033[0m")
    
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
  ## Test the model with test data
  my_test = ['no way I am giving them my money', 
             'send money to african prince',
             'please send a million dollar to help poor africans',
             'I don\'t know u and u don\'t know me. Send CHAT and money',
             'Send me your email address and I will send youa million dollars'
            ]

  # test_predictions = mnb_clf_model.predict(vect.transform(my_test))
  test_predictions = mnb_clf_model.predict(vect.transform(X_test))
  


  #
  ## Calculate the AUC:
  auc = roc_auc_score(y_test, test_predictions)
  print(f'| \033[1m AUC score is:  {auc}\033[0m')

  
  return(mnb_clf_model)

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
def answer_four():
  pass
  #return()


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
def answer_five():
  pass
  #return()



#
## Question 6
##-----------------------------------------------------------------------------
## What is the average length of documents (number of characters) for not spam 
## and spam documents?
##
## Return: This function should return a tuple (average length not spam, average 
## length spam).
##
def answer_six():
  pass
  #return()



#
## Question 7
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Tfidf Vectorizer 
## ignoring terms that have a document frequency strictly lower than 5.
## 
## Using this document-term matrix and an additional feature, the length of 
## document (number of characters), fit a Support Vector Classification model 
## with regularization C=10000. Then compute the area under the curve (AUC) 
## score using the transformed test data.
##
##
## Return:This function should return the AUC score as a float.
##
def answer_seven():
  pass
  #return()



#
## Question 8
##-----------------------------------------------------------------------------
## What is the average number of digits per document for not spam and spam 
## documents?
##
## Return: This function should return a tuple (average # digits not spam, 
##         average # digits spam).
##
def answer_eight():
  pass
  #return()
  



#
## Question 9
##-----------------------------------------------------------------------------
## Fit and transform the training data X_train using a Tfidf Vectorizer 
## ignoring terms that have a document frequency strictly lower than 5 and
##  using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams).
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
def answer_nine():
  pass
  #return()
  



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
def answer_ten():
  pass
  #return()  
  



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
##  * Tnumber of digits per document
##  * Tnumber of non-word characters (anything other than a letter, digit or 
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
def answer_eleven():
  pass
  #return()  
  
  

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
  mnb_clf_model = answer_three(X_train, X_test, y_train, y_test)
