#
#
import inspect
import numpy as np
import pandas as pd
import pprint

from pprint import pformat
from pprint import PrettyPrinter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tabulate import tabulate

#
## References:
#  https://blog.devgenius.io/how-to-easily-print-and-format-tables-in-python-18bbe2e59f5f
#  https://www.educba.com/python-print-table/
#  https://learnpython.com/blog/print-table-in-python/
#

#
## Objective: 
##  1) Calculate the TF-IDF for each sentance
##  2) Calculate the TF-IDF for the entire corps
##  3) Compare #1 and #2 and inspect how the differ
##

#
## 
##
##
def load_data_file(file_name='../../data/spam-dummy.csv'):
  
  my_pp = pprint.PrettyPrinter(indent=4)
  my_pp.pprint(stuff)
  
  #
  ## Print function header
  this_function_name = inspect.currentframe().f_code.co_name
  print(f"<---- {this_function_name}---->")
  
  
  #file_name = '../../data/spam.csv'
  file_name = '../data/spam-dummy.csv'

  #
  ## Create a data-frame from the CSV file
  #
  spam_data_df = pd.read_csv(file_name)

  spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)
  X_train, X_test, y_train, y_test = train_test_split(spam_data_df['text'], 
  spam_data_df['target'],
  test_size=0.0001, 
  random_state=0)
  
  # print(f'. Xtrain data-type is: {type(X_train)}')
  # my_pp.pprint(X_train)
  
  print(tabulate(X_train, headers='keys', tablefmt='psql'))
  # my_pp.pprint(tabulate(X_train))
  
  return(X_train, X_test, y_train, y_test)

  

if __name__ == "__main__":

  
  stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
  stuff.insert(0, stuff[:])
  my_pp = pprint.PrettyPrinter(indent=10)
  # my_pp.pprint(stuff)
  
  file_name='../../data/spam-dummy.csv'
  X_train, X_test, y_train, y_test = load_data_file(file_name)
  # X_train_list = describe_data(X_train)
  # calculate_tfidf(X_train_list)

