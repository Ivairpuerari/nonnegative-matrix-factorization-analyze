import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

FileWithStopWords = 'C:/Users/Ivaai/Documents/clean/clean_admissionLife.txt'

# 'C:/Users/Ivaai/Documents/clean/clean_admissionLife.txt'

# 'C:/Users/Ivaai/Documents/clean_admissionDeath.txt'

with open(FileWithStopWords, 'r',  encoding='UTF-8') as fr:
    data = [line.replace('\n', '') for line in fr]

print(len(data))

tfidf_vectorizer = TfidfVectorizer(min_df=10, stop_words='english', max_features=20000,
                                   analyzer='word')

tfidf = tfidf_vectorizer.fit_transform(data)


tfidf_feature_names = tfidf_vectorizer.get_feature_names()

print('size %d of terms' % len(tfidf_feature_names))

n_topics = 11

nmf = NMF(n_components=n_topics, random_state=1,
          alpha=0.1, solver="mu")

W = nmf.fit_transform(tfidf)

H = nmf.components_


col1 = 'topic'
col2 = 'top_words'
dct = {col1: [], col2: []}
no_top_words = 10

for topic_id, topic in enumerate(H):
    dct[col1].append(str(topic_id))
    dct[col2].append(" ".join([tfidf_feature_names[i]
                               for i in topic.argsort()[:-no_top_words - 1:-1]]))

topic_word = pd.DataFrame.from_dict(dct)

print(topic_word)

topic_word.to_csv('topics_clean.csv')
