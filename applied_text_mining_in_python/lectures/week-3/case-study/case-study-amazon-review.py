#
## This script will attempt to classify amazon unlocked
## smart phones
#
#*                                                                             *

#
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

#
## Lab URL:
##    https://www.coursera.org/learn/python-text-mining/ungradedLab/knFQT/case-study-sentiment-analysis/lab
#
## Lab explanation
##  https://www.coursera.org/learn/python-text-mining/lecture/MJ7g3/demonstration-case-study-sentiment-analysis

#
## Already implemented:
##  https://github.com/Brucewuzhang/Applied-text-mining-in-python/find/master

#
## Reference:
# https://www.educative.io/answers/countvectorizer-in-python
# 
#
## Read the reviews into a data-frame
df = pd.read_csv('data/Amazon_Unlocked_Mobile.csv')

#*                                                                             *
#
#
## Print the head of the data-frame, which consists of:
#
# Product |  Brand  |  Price |  Rating  |  Reviews |  Review |
# name    |  name   |        |          |          |   Votes |
#--------------------------------------------------------------
#
# For our purposes, we'll be focusing on the following columns:
#
#  - Rating
#  - Reviews
#
#*                                                                             *

print(f'\n  Data-frame head: {df.head()}')
print(f'  Length of data frame: {len(df)}')

#
## Drop any rows with missing values
df.dropna(inplace = True)


#
## Drop columns with ratings equal to 3, we do this by creating a new data 
## frame, consisting of only rows, which have a rating different than 3
df=df[df['Rating']!= 3]


print(f'\n  Length of the new data frame: {len(df)}')

#*                                                                             *
##
## 
## Create a new column "Positive Review" that will serve as our target for our
## model, where any reviews that were rated more than 3 will be encoded as a 1,
## indicating it was positively rated. Otherwise, it'll be encoded as a 0,
## indicating it was not positively rated.
##
## np sytax below:
##   
##   np.where(condition, value_if_true, value_if_false)

df['Positively Rated']= np.where(df['Rating']>3, 1, 0)
print(f'\n  Data-frame head after review ratings: {df.head()}')

#
## Display the mean of the positively rated reviews:
print(f"\n  Positively Rated: {df['Positively Rated'].mean()}")

#
## mean of the positively rated class is: 0.7482686025879323
## this indicates that our data is imbalanced. This tells us that 74% of 
## the data is positively reviewed. If the mean was 0.5, then we could say 
## it was balanced.

#
## Split our data into training and test sets using:
##
##  1) Reviews
##  2) Positively rated columns
##
## random_state - Controls the shuffling applied to the data before
## applying the split
##
## This creates two pairs of data sets - X_train, X_test, y_train, y_test
##
X_train, X_test, y_train, y_test = train_test_split(
     df['Reviews'],
     df['Positively Rated'],
     random_state=0
   )

# print(f'\n  Lenght of X_train is: {len(X_train)}')

print('\n  X_train first entry:\n\t \033[1m', X_train.iloc[0], '\033[0m')

#
## shape here represent number of documents in the training set
print('\n\n  X_train shape: \033[1m', X_train.shape, '\033[0m')

#
## Shape returned is 23,1207

#
##              COUNT VECTORIZER EXPLANATION
##
## Next we need to convert our train/test data into numeric representation 
## that scikit-learn can use. The bag-of-words approach is simple and 
## commonly used way to represent text for use in machine learning, which
## ignores structure and only counts how often each word occurs. 
##
## CountVectorizer allows us to use the bag-of-words approach by converting 
## a collection of text documents into a matrix of token counts. First
## everything is converted to lower-case and a vocabulary is built using
## these tokens
##
## Fitting the CountVectorizer consists of the tokenization of the trained 
## data and building of the vocabulary. Example below:
##
#
#*                                                                             *
##
##
## text = ['Hello my name is John Doe. CS is my love'] 
##
## and its matrix representation:
##
##  Row | hello | my | name | is | john | doe | cs | love
##  ----+-------+----+------+----+------+-----+----+------
##   0  |   1   |  2 |  1   |  2 |   1  | 1   | 1  |  1
##
## We are tallying up the tokens and their occurances. Only tokens that 
## appear twice are: is, my
## 
## 

#
## tokenize and build vocab - further explanation above ^^
vect = CountVectorizer().fit(X_train)

#
##                        VOCABULARY
##
## Next, we inspect the vocabulary by using the get_feature_names_out 
## method.  This vocabulary is built on any tokens that occurred in the 
## training data.
## Looking at every 2,000th feature, we can get a small sense of what the 
## vocabulary looks like. We can see it looks pretty messy, including words 
## with numbers as well as misspellings.
##
print(f'\n  Features of matrix:\n\t {vect.get_feature_names_out()[::2000]}')




#
##                       Number of Features
##
## We can check how many features we are working with, by checking
## the length of get_feature_names
vector_feature_names_len = len(vect.get_feature_names_out())
print(f'\n  Number of features in the matrix:\033[1m {vector_feature_names_len}', ' \033[0m')



#
##                   Document Term Matrix - bag-of-words
##
## Now we transform the documents in X_train to a document term matrix
## using transform method. This gives us a bag-of-word representation of 
## X_train
#
## This representation is stored in a SciPy sparse matrix, where each row
## corresponds to a document and each column a word from our training
## vocabulary. 
## 
## Reference: 
##  https://analyticsindiamag.com/a-guide-to-term-document-matrix-with-its-implementation-in-r-and-python/
##
## The entries in this matrix are the number of times each word appears 
## in each document. Example below:
##
#  +-------+--------------------------------------------------+
#  | Index |  Sentences                                       |  
#  +-------+--------------------------------------------------+
#  |  1    | I love football                                  |
#  +-------+--------------------------------------------------+
#  |  2    | Messi is a great football player                 |
#  +-------+--------------------------------------------------+
#  |  3    | Messi has won seven Ballon D'Or awards           |
#  +-------+--------------------------------------------------+
#
#  Respective Document-Term Matrix would look like the following:
#
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+
#  |Doc| I | a | is | has | won | D'or | seven | love | messi | player | ballon | award | great | football |
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+
#  | 1 | 1 |   |    |     |     |      |       |   1  |       |        |        |       |       |    1     |
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+
#  | 2 |   | 1 |  1 |     |     |      |       |   1  |       |   1    |        |       |   1   |    1     |
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+
#  | 3 |   |   |    |  1  |  1  |   1  |   1   |      |   1   |        |    1   |   1   |       |          |
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+
#  |   |   |   |    |     |     |      |       |      |       |        |        |       |       |          |
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+
#  |Tot| 1 | 1 |  1 |  1  |  1  |   1  |   1   |  2   |   1   |        |    1   |   1   |       |    2     |
#  +---+---+---+----+-----+-----+------+-------+------+-------+--------+--------+-------+-------+----------+

X_train_vectorized = vect.transform(X_train)
# print(f'\n  Training data vectorized: {X_train_vectorized}')


print(f'\n  Type of X_train_vectorized:\033[1m {type(X_train_vectorized)}', ' \033[0m')


#
## Convert sparse matrix to pandas data frame
## Reference:
##   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sparse.from_spmatrix.html

#
## Convert the sparse matrix to data-frame and write the data-frame
## to file
# X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_vectorized)
# X_train_df.to_csv('/tmp/blah/count_vectorizer.csv')

#
## Save the sparse matrix to file:
# sparse.save_npz("yourmatrix.npz", X_train_vectorized)

#*                                                                            *

#
## Now we use Logistic Regression to train our model
#
## Good video on Logistic Regressions:
#    https://www.youtube.com/watch?v=yIYKR4sgzI8
##
## LogisticRegression - which works well for high dimensional sparse data.
## 
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

''' 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
   intercept_scaling=1, max_iter=300000, multi_class='ovr', n_jobs=1, 
   penalty='12', random_state=None, solver='liblinear', tol=0.0001,
   verbose=0, warm_start=False
   )

# LogisticRegression(solver='liblinear', dual=False, verbose=0)
# log_reg = LogisticRegression(solver='lbfgs', class_weight='balanced', dual=False, max_iter=10000)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
   intercept_scaling=1, multi_class='ovr', n_jobs=1, 
   penalty='12', random_state=None, solver='liblinear', tol=0.0001,
   verbose=0, warm_start=False
   )

log_reg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)

'''
   
#*                                                                            *



#
## Next, we'll make predictions using X_test and compute the area under 
## the curve score. We'll transform X_test using our vectorizer that was 
## fitted to the training data. 
##
## AUC - area under the curve, provides an aggregate measure of performance 
## across all possible classification thresholds. One way of interpreting AUC 
## is as the probability that the model ranks a random positive example more 
## highly than a random negative example
##
## Note that any words in X_test that didn't appear in X_train will just be 
## ignored.
##
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


print(f'|  \033[1m predictions are: {predictions} \033[0m')
#
## predictions are: ['ham' 'ham' 'ham'] 
print("-------------------------------------------------------------")


print(f'|  \033[1m predictions type: {type(predictions)} \033[0m')
print("-------------------------------------------------------------")

print(f'|  \033[1m y_test length: {len(list(y_test))} \033[0m')
print("-------------------------------------------------------------")

print(f'|  \033[1m y_test columns: \n{list(y_test)} \033[0m')
print("-------------------------------------------------------------")

print(f'|  \033[1m y_test description: \n{y_test.describe()} \033[0m')
print("-------------------------------------------------------------")

print(f'|  \033[1m y_test head: {y_test.head()} \033[0m')
print("-------------------------------------------------------------")



# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

#
## Find the 10 smallest and 10 largest coefficients
## The 10 largest coefficients are being indexed using [:-11:-1] 
## so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

#*                                                                            *

#################################### TFIDF ####################################
# 
# TFIDF - Term Frequence Inverse Document Frequence. 
# 
## Tf–idf, or Term frequency-inverse document frequency, allows us to weight 
## terms based on how important they are to a document.
#
## High weight is given to terms that appear often in a particular document, 
## but don't appear often in the corpus. Features with low tf–idf are either 
## commonly used across all documents or rarely used and only occur in long 
## documents.
#
## Features with high tf–idf are frequently used within specific documents, but
## rarely used across all documents.
#
## Fit the TfidfVectorizer to the training data specifiying a minimum document 
## frequency of 5 - here we'll pass in min_df = 5, which will remove any words 
## from our vocabulary that appear in fewer than five documents.
#
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))
                                    
#*                                                                            *

################################### n-gram ####################################
# 
## Our current model sees both of the following as negative reviews:
# 
##  'not an issue, phone is working'
##  'an issue, phone is not working'
#
## One way we can add some context is by adding sequences of word features known 
## as n-grams. For example, bigrams, which count pairs of adjacent words, could 
## give us features such as is working versus not working. And trigrams, which 
## give us triplets of adjacent words, could give us features such as not an issue.
#
## Fit the CountVectorizer to the training data specifiying a minimum 
## document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))                                    