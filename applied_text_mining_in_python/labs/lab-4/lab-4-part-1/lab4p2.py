#
## Lab URL:
## https://www.coursera.org/learn/python-text-mining/programming/2qbcK/assignment-4/instructions
##
##
#
## Additional Resource/Solution:
##  https://github.com/umer7/Applied-Text-Mining-in-Python/blob/master/Assignment%2B4.ipynb
##



import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer


# 
## Part 2 - Topic Modelling
#------------------------------------------------------------------------------
# For the second part of this assignment, you will use Gensim's LDA (Latent 
# Dirichlet Allocation) model to model topics in newsgroup_data. You will first
# need to finish the code in the cell below by using 
# gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters
# on the corpus, and save to the variable ldamodel. Extract 10 topics using 
# corpus and id_map, and with passes=25 and random_state=34.

def load_newsgroups_data():

  # Load the list of documents
  with open('data/newsgroups', 'rb') as f:
      newsgroup_data = pickle.load(f)
  
  # Use CountVectorizor to find three letter tokens, remove stop_words, 
  # remove tokens that don't appear in at least 20 documents,
  # remove tokens that appear in more than 20% of the documents
  vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                         token_pattern='(?u)\\b\\w\\w\\w+\\b')
  # Fit and transform
  X = vect.fit_transform(newsgroup_data)
  
  # Convert sparse matrix to gensim corpus.
  corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
  
  # Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
  id_map = dict((v, k) for k, v in vect.vocabulary_.items())
  
  
  # Use the gensim.models.ldamodel.LdaModel constructor to estimate 
  # LDA model parameters on the corpus, and save to the variable `ldamodel`
  
  # Your code here:
  ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=10, 
           id2word=id_map, passes=25, random_state=34)



# 
# lda_topics
# Using ldamodel, find a list of the 10 topics and the most significant 10 
# words in each topic. This should be structured as a list of 10 tuples where 
# each tuple takes on the form:
# 
# (9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 
# 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 
# 0.014*"center" + 0.014*"sci"')
#
# for example.
#
# This function should return a list of tuples.
#
def lda_topics():
    
    # Your Code Here
    
    return lda.print_topics()
    

#
# topic_distribution
# For the new document new_doc, find the topic distribution. Remember to use 
# vect.transform on the the new doc, and Sparse2Corpus to convert the sparse 
# matrix to gensim corpus.
#
# This function should return a list of tuples, where each tuple is (#topic, 
# probability)

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]


def topic_distribution():
    
    new_doc_vectorized = vect.transform(new_doc)
    doc2corpus = gensim.matutils.Sparse2Corpus(new_doc_vectorized, documents_columns=False)
    
    return list(ldamodel.get_document_topics(doc2corpus))[0]
    
    
#
## topic_names
#
# From the list of the following given topics, assign topic names to the topics
# you found. If none of these names best matches the topics you found, create a
# new 1-3 word "title" for the topic.
#
# Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers
# & IT, Sports, Business, Society & Lifestyle, Religion, Education.
#
# This function should return a list of 10 strings.
#
#
def topic_names():
    
    topic_names = ['Health', 'Automobiles', 'Government', 'Travel', 'Computers & IT', 'Sports', 'Business', 'Society & Lifestyle', 'Region', 'Education']
    topics = lda_topics()
    results = []
    for _, dis in topics:
        print(dis)
        similarity = []
        for topic in topic_names:
            similarity.append(document_path_similarity(dis, topic))
        best_topic = sorted(zip(similarity, topic_names))[-1][1]
        results.append(best_topic)
    return ['Education', 'Business', 'Automobiles', 'Religion', 'Travel', 'Sports', 'Health', 'Society & Lifestyle', 'Computers & IT', 'Science']


if __name__ == "__main__":
  df = load_newsgroups_data()
  #
  ## There are two columns in the CSV:
  ##  1) Text
  ##  2) Target
  ##
  # print(f'  Length of data frame is: {len(df)}')
  # print(f'  Head of the data frame: {df.head()}')

