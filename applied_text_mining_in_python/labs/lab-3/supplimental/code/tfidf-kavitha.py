import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
import re
from varname import nameof
from tabulate import tabulate
from datetime import datetime

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

  '''
  # print("\033[1m ")
  print("+------------------------------------------------------------")
  print("|  \033[1m --- load_data --- \033[0m")
  print("+------------------------------------------------------------")
  '''
  
  # print("|  before filtering")
  # print(spam_data_df.head())
  # pprint(spam_data_df.head(), indent=3)
  # pprint(f'Spam head data type is: {type(spam_data_df.head())}, indent=3')
  # print(tabulate(spam_data_df, headers='keys', tablefmt='psql'))

  
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
  print("|  \033[1m --- add_feature --- \033[0m")
  print("+------------------------------------------------------------")
  
  # print(f'|  Adding feature {feature_to_add} to {X}')
  print(f'|  Adding feature feature_to_add to X')
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  print(f'|  \033[1m nameof(X) is of type: {type(X)} \033[0m')
  print(f'|  \033[1m X shape(rows/columns) before modification is: {X.shape} \033[0m')
  '''

  X_modified = hstack([X, csr_matrix(feature_to_add).T], 'csr')
  
  # print(f'|  \033[1m X_modified shape(rows/columns) after modification is: {X_modified.shape} \033[0m')

  
  return(X_modified)

#
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
  # print(spam_data_df)
  spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)

  X_train, X_test, y_train, y_test = train_test_split(spam_data_df['text'], 
                                                      spam_data_df['target'],
                                                      test_size=0.3, 
                                                      random_state=0)  
  return(X_train, X_test, y_train, y_test)
  

##-----------------------------------------------------------------------------


def answer_eleven():
  
  ''' 
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 11 \033[0m")
  print("+------------------------------------------------------------")
  ''' 

  #
  ## df consists of two columns:
  ##  - text 
  ##  - target 
  

  # print("| \033[1m Vectorizing Training Data \033[0m")

    
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
  # print(f"| \033[1m Vectorizer shape(rows/columns): {X_train_vectorized.shape} tokens \033[0m ")
  # print(vector.toarray())
  
  # print(f"|  Vectorizing shape after fit_transform: {vectorizer.shape}")

  
  #
  ## From X_train(pandas data series) get additional features
  (X_train_doclen, X_train_numdigits, X_train_nonalpha) = feature_extractor(X_train) 
  
  #
  ## Iterate over the data returned by feature_extractor and add each
  ## individual feature series to X_train_vectorized
  for feature in (X_train_doclen, X_train_numdigits, X_train_nonalpha):
    # print(f"|  nameof(feature) is of type: {type(feature)}")
    # print(f"|  Appending feature {nameof(feature)} to X_train_vectorized")
    X_train_vectorized = add_feature(X_train_vectorized, feature)
    # print(f"|  nameof(X_train_vectorized) is of type: {type(X_train_vectorized)}")

    
  # print("| \033[1m Vectorizing Testing Data \033[0m")

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

  # print(f"|  Document Term Matix is of type: {type(X_test_vectorized)}")
  # print(f"|  Document Term Matix has the shape(Rows x Column): {X_test_vectorized.shape}")
  
  
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
  '''

  # print(f"|  feature_names_array: {feature_names_array}")
  
  #
  ## push new feature names into feature_names array
  feature_names_array = np.append(feature_names_array, 'length_of_doc')
  feature_names_array = np.append(feature_names_array, 'digit_count')
  feature_names_array = np.append(feature_names_array, 'non_word_char_count')
  print(f"|  *** After addition - feature_names_array has the shape: {feature_names_array.shape}")

   
  coef_rankings = pd.DataFrame(
    classifier.coef_[0].argsort(),
    columns = ["Coefficient"],
    index = feature_names_array,
  )
  
  
  # print(f"|  Coefficient rankings: {coef_rankings}")
  # print(tabulate(coef_rankings.sort_values(by=['Coefficient']), headers='keys', tablefmt='psql'))
  

  un_labeled = ['Ford F150 Lightening is a game changer for Ford.',
                'According to Doug DeMouro, this is the Model 3 of Pickups',
                'A free Tesla Model 3 awaits you!',
                'call 1-800-buy-EVNOW to get your very own Electric Truck',
                'PRIVATE! you have free 800 un-redeemed',
                'GMs Hummer EV is a monstrosity,'
                'RAM REV is a gas range extender',
                'Call 1-800-994-2343 for free megazine subscription'
               ]
  
  #
  ## Create a sparse-matrix from X_test 
  un_labeled_vectorized = vectorizer.transform(un_labeled)
  
  
  ## From X_train(pandas data series) get additional features
  (un_labeled_doclen, un_labeled_numdigits, un_labeled_nonalpha) = feature_extractor(un_labeled) 
  

  un_labeled_vectorized = add_feature(un_labeled_vectorized, un_labeled_doclen)
  un_labeled_vectorized = add_feature(un_labeled_vectorized, un_labeled_numdigits)
  un_labeled_vectorized = add_feature(un_labeled_vectorized, un_labeled_nonalpha)
  
  predictions = classifier.predict(un_labeled_vectorized)
  print(f"|  nameof(predictions) is of type: {type(predictions)}")
  
  # for (string, prediction) in (un_labeled, prediction):
  # for (prediction) in (prediction):
  #  for (string) in (un_labeled):
  for (prediction, string) in zip(predictions, un_labeled):
    print("+-----+----------------------------------------------------")
    print(f"|  {prediction}  | {string} ")
  print("+-----+----------------------------------------------------")

  # feature_names = np.array(vectorizer.get_feature_names_out())
  feature_names = vectorizer.get_feature_names_out()
  feature_names_array = np.array(feature_names)
  
  ''' 
  print(f"|  *** feature_names is of type: {type(feature_names)}")
  print(f"|  *** feature_names_array has the shape: {feature_names_array.shape}")
  print(f"|  *** feature_names_array is of type: {type(feature_names_array)}")
  # print(f"|  feature_names_array: {feature_names_array}")
  '''
  
  #
  ## push new feature names into feature_names array
  feature_names_array = np.append(feature_names_array, 'length_of_doc')
  feature_names_array = np.append(feature_names_array, 'digit_count')
  feature_names_array = np.append(feature_names_array, 'non_word_char_count')
  
  
   
  coef_rankings = pd.DataFrame(
    classifier.coef_[0].argsort(),
    columns = ["Coefficient"],
    index = feature_names_array,
  )

  #
  ## Add a column header to the index column
  coef_rankings = coef_rankings.rename_axis('Feature')
    
  print(tabulate(coef_rankings.sort_values(by=['Coefficient']), headers='keys', tablefmt='psql'))
  
  
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=
#                         MAIN Starts below....
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=

if __name__ == "__main__":
  

  # file_name = "../data/spam-dummy.csv"
  file_name = "../data/spam-short.csv"
  # file_name = "data/SMSSpamCollection"
  # file_name = "../data/spam-dummy-longer.csv"
  
  df = load_data(file_name)
  #
  ## There are two columns in the CSV:
  ##  1) Text
  ##  2) Target
  ##
  # print(f'  Length of data frame is: {len(df)}')
  # print(f'  Head of the data frame: {df.head()}')
  
  #
  ## Split data into training and test
  X_train, X_test, y_train, y_test = split_training_data(df)
  answer_eleven()
