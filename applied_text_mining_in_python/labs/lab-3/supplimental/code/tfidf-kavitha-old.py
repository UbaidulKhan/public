import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
import re
from varname import nameof

##-----------------------------------------------------------------------------
#
def add_feature(X, feature_to_add):
  # print(f"  add_feature - feature to add: {feature_to_add}")
  # print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--==-=-=-=-=-=")
  # X_modified = hstack([X, csr_matrix(feature_to_add).T], 'csr')
  
  # new_dtm = np.concatenate((X, feature_to_add.array), axis=1)
  
  print(f"  X is of type: {type(X)}")
  
  # return(X_modified)

##-----------------------------------------------------------------------------
#
def feature_extractor(series_data):

  series_doc_len = []
  series_digits = []
  series_non_alphas = []
  
  entry = 0
  for (idx, text) in enumerate(series_data):
    
    text_length = (len(text))

    text_digits = sum(c.isdigit() for c in text)
    
    text_non_alphas =  re.findall(r'\W+', text)
    text_non_alphas_count =  len(text_non_alphas)

    series_doc_len.append(text_length)
    series_digits.append(text_digits)
    series_non_alphas.append(text_non_alphas_count)

  series_doc_len_series = pd.Series(series_doc_len)
  series_digits_series = pd.Series(series_digits)
  series_non_alphas_series = pd.Series(series_non_alphas)
  

  series_doc_len_renamed = series_doc_len_series.rename('length_of_doc')
  series_digits_renamed =  series_digits_series.rename('digit_count')
  series_non_alphas_renamed = series_non_alphas_series.rename('non_word_char_count')

  
  return(series_doc_len_renamed, series_digits_renamed, series_non_alphas_renamed)  


##-----------------------------------------------------------------------------
#
def load_csv_data(file_name):
  
  spam_data_df = pd.read_csv(file_name)
  spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)

  X_train, X_test, y_train, y_test = train_test_split(spam_data_df['text'], 
                                                      spam_data_df['target'],
                                                      test_size=0.3, 
                                                      random_state=0)  
  return(X_train, X_test, y_train, y_test)
  

##-----------------------------------------------------------------------------

file_name = "../data/spam-dummy.csv"
# file_name = "../data/spam-short.csv"

X_train, X_test, y_train, y_test = load_csv_data(file_name)
print(f"  X_train is: {X_train}")
print("--------------------------------------------------------")
print(f"  X_test is: {X_test}")
print("--------------------------------------------------------")
print(f"  y_train is: {y_train}")
print("--------------------------------------------------------")
print(f"  y_test is: {y_test}")
 
vectorizer = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb')
X_train_vectorized = vectorizer.fit_transform(X_train)
(X_train_doclen, X_train_numdigits, X_train_nonalpha) = feature_extractor(X_train) 

for feature in (X_train_doclen, X_train_numdigits, X_train_nonalpha):
  X_train_vectorized = add_feature(X_train_vectorized, feature)

X_test_vectorized = vectorizer.transform(X_test)
(X_test_doclen, X_test_numdigits, X_test_nonalpha) = feature_extractor(X_test) 

for feature in (X_test_doclen, X_test_numdigits, X_test_nonalpha):
  X_test_vectorized = add_feature(X_test_vectorized, feature)

classifier = LogisticRegression(C=100, solver='liblinear')
classifier.fit(X_train_vectorized, y_train)
y_predicted = classifier.predict(X_test_vectorized)


feature_names = np.array(vectorizer.get_feature_names_out())

for feature_name in feature_names: 
  print(f"  Inspecting feature: {feature_name}")

  if(feature_name == 'length_of_doc'):
    print(f'  Feature name: {feature_name} has been found')
    
  elif(feature_name == 'digit_count'):
    print(f'  Feature name: {feature_name} has been found')
  
  elif(feature_name == 'non_word_char_count'):
    print(f'  Feature name: {feature_name} has been found')
  
  
''' 

for feature_name in feature_names:
  print(f"  Feature name is: {feature_name}")

sorted_coef_index = classifier.coef_[0].argsort()
smallest = feature_names[sorted_coef_index[:10]]
largest = feature_names[sorted_coef_index[:-11:-1]]
'''