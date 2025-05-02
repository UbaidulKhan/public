
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

file_name = '../../data/spam.csv'
spam_data_df = pd.read_csv(file_name)

spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)
X_train, X_test, y_train, y_test = train_test_split(spam_data_df['text'], 
                                                      spam_data_df['target'],
                                                      test_size=0.3, 
                                                      random_state=0)

#
## Describe the X_train and get column names
X_train_list = X_train.tolist()
print(f'X_train length: {len(X_train_list)}')


X_train_list_one = X_train_list[0:2]
print(f'X_train_list contains: \n{X_train_list_one}')


# 
#
## Describe the X_train and get column names
# X_train_np_array = np.array(X_train)
# print(f'X_train_list contains: {X_train_np_array[0:2]}')



# 
## Create a vecorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True) 


# 
## Fit and Transform the training data 
tfidf_vectorizer_fit = tfidf_vectorizer.fit(X_train_list)
# tfidf_vectorizer_vectors = tfidf_vectorizer.inverse_transform(X_train)
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(X_train_list)



# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()
# print(f'Feature name type is: {type(feature_names)}')

#
## Print all the features
for name in feature_names:
  print(f'Feature names: {name}\n')

# print(f'Feature names length: {len(feature_names)}')
# print(f'Feature names: {feature_names}')


# Create a dense 
tfidf_vectorizer_dense = tfidf_vectorizer_vectors.todense()
tfidf_vectorizer_dense_transverse = tfidf_vectorizer_dense.T
# print(f'Data type is: {type(tfidf_vectorizer_dense_transverse)}')

# tfidf_dense_list = tfidf_vectorizer_dense.tolist()
# print(f'tfidf_dense_list: {tfidf_dense_list}')


#
## This works and creates a data-frame
df = pd.DataFrame(tfidf_vectorizer_dense_transverse, 
                  index=feature_names)
                  # index=feature_names).reset_index() 
                  # columns=["tfidf"])
                  
print(f'Data Frame head: {df[100:120]}')

#
## Data Frame information
print(f'Data Frame Description: {df.describe()}')
print(f'Data Frame Description: {len(df.columns)}')

# for col in df.columns:
#    print(col)
                  


