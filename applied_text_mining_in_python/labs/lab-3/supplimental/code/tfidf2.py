
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


#instantiate CountVectorizer() 
cv=CountVectorizer() 

# this steps generates word counts for the words in your docs 
word_count_vector=cv.fit_transform(X_train)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)

# print idf values 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"]) 
# sort ascending 
df_idf.sort_values(by=['idf_weights'])

print(df_idf)


# count matrix 
count_vector=cv.transform(X_train) 
# tf-idf scores 
tf_idf_vector=tfidf_transformer.transform(count_vector)


feature_names = cv.get_feature_names_out() 
#get tfidf vector for first document 
first_document_vector=tf_idf_vector[0] 
#print the scores 
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 

print(df.sort_values(by=["tfidf"],ascending=False))


