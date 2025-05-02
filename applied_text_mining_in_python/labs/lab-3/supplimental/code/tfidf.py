
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 

file_name = '../../data/spam.csv'
spam_data_df = pd.read_csv(file_name)

spam_data_df['target'] = np.where(spam_data_df['target']=='spam',1,0)
X_train, X_test, y_train, y_test = train_test_split(spam_data_df['text'], 
                                                      spam_data_df['target'],
                                                      test_size=0.3, 
                                                      random_state=0)

# settings that you use for count vectorizer will go here 
tfidf_vectorizer=TfidfVectorizer(use_idf=True) 


# just send in all your docs here 
vectors = tfidf_vectorizer.fit_transform(X_train)


# Get features
feature_names = tfidf_vectorizer.get_feature_names_out()



#place tf-idf values in a pandas data frame 
tfidf_dense = vectors.todense() 
tfidf_dense_list = tfidf_dense.tolist()

print(f'List length is: {len(tfidf_dense_list)}')
# print(f'List head is: {tfidf_dense_list[0]}')


df = pd.DataFrame(tfidf_dense_list, 
                  index=feature_names,
                  columns=["tfidf"]) 

print(f' Data Frame description: {df.describe()}')
print(f' Data Frame head: {df.head()}')


df = pd.DataFrame(tfidf_dense_list, 
                  index=tfidf_vectorizer.get_feature_names_out(), 
                  columns=["tfidf"]) 
print(df.sort_values(by=["tfidf"],ascending=False))

