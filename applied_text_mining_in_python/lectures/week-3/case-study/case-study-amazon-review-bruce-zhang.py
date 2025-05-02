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

#
## Reference:
#
# Bruce Zhang Implementation
# https://github.com/Brucewuzhang/Applied-text-mining-in-python/blob/master/Text%20Mining%EF%BC%9AAmazon%20unlocked%20mobile%20review%20.ipynb 
#
#

df = pd.read_csv('Amazon_Unlocked_Mobile.csv')
df.head()

len(df)

df.dropna(inplace = True)

len(df)

#regard rating 3 as neutral and drop it
df=df[df['Rating']!= 3]
len(df)

df['Positive Review']= df['Rating']>3


df['Positive Review'] = df['Positive Review'].apply(int)


X = df['Reviews']
y = df['Positive Review']

y.mean()

#
## Use CountVectorizer and LogisticRegression
##
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42)

vect = CountVectorizer().fit(X_train)

X_train_vectorized = vect.transform(X_train)


lr_clf = LogisticRegression()
lr_clf.fit(X_train_vectorized,y_train)
y_pred = lr_clf.predict(vect.transform(X_test))
print("AUC score: ", roc_auc_score(y_pred,y_test))

lr_clf.predict(vect.transform(['good','not good','bad','not bad']))


#
## Use 2-gram
##
lr_clf1 = LogisticRegression()
vect = CountVectorizer(ngram_range=(1,2)).fit(X_train)
X_train_vect = vect.transform(X_train)
lr_clf1.fit(X_train_vect,y_train)
y_pred1= lr_clf1.predict(vect.transform(X_test))
print("AUC score using 2-gram :", roc_auc_score(y_test,y_pred1))

lr_clf.predict(vect.transform(['good','not good','bad','not bad']))

vect = CountVectorizer(ngram_range=(1,2),min_df = 5).fit(X_train)

X_train_vect2 = vect.transform(X_train)
lr_clf.fit(X_train_vect2,y_train)
y_pred = lr_clf.predict(vect.transform(X_test))
print("AUC score using 2-gram and min_df =5 :", roc_auc_score(y_test,y_pred))

print("AUC score using 2-gram and min_df =5 :", roc_auc_score(y_train,lr_clf.predict(X_train_vect2)))



