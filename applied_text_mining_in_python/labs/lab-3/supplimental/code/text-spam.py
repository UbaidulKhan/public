
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time

input_file_name = 'data/SMSSpamCollection'
output_file_name = 'data/spam-text.txt'


spam_data_df = pd.read_csv(input_file_name, sep="\t")

# 
## Create a list to append all the lines from the input file  
docs = []

# print(spam_data_df.head())

#
## Remove the first column - target
column_to_move = spam_data_df.pop("target")
# spam_data_df.insert(0, "text", column_to_move)
# print(spam_data_df.head())

for index, row in spam_data_df.iterrows():
  docs.append(row['text'])


# settings that you use for count vectorizer will go here 
tfidf_vectorizer=TfidfVectorizer() 


# just send in all your docs here 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)
# tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(df)

 
# get the first vector out (for the first document) 
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0] 

# place tf-idf values in a pandas data frame 
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), 
                  index=tfidf_vectorizer.get_feature_names_out(), 
                  columns=["tfidf"]
                  ) 
                  
df.sort_values(by=["tfidf"],ascending=True)

#
## Display all rows of the data-frame
pd.set_option('display.max_rows', None)

print(df)


