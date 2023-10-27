import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
import re
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    sim_score = list(enumerate(cosine_sim[-1]))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    sim_score = sim_score[1:11]
    movie_idx = [i[0] for i in sim_score]
    recommend_movie_list = df_reviews.iloc[movie_idx, 0]
    return recommend_movie_list

df_reviews = pd.read_csv('./preprocessing.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle','rb') as f:
    Tfidf = pickle.load(f)

# print(df_reviews.iloc[2068,0])
# cosine_sim = linear_kernel(Tfidf_matrix[2068], Tfidf_matrix)
# print(cosine_sim)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)

# 키워드 기반 추천
# embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
# keyword = '잔인'
# sim_word = embedding_model.wv.most_similar(keyword, topn=10)
# print(sim_word)
# words = [keyword]
# for word, _ in sim_word:
#     words.append(word)
# print(words)
#
# sentence = []
# count = 10
# for word in words:
#     sentence = sentence + [word]*count
#     count -= 1
# sentence = ' '.join(sentence)
# print(sentence)
# sentence_vec = Tfidf.transform([sentence])
# cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)

# 문장 기반 추천

okt = Okt()

sentence = '풋풋한 첫사랑 같은 영화'
sentence = re.sub('[^가-힣]',' ',sentence)
tokened_sentence = okt.pos(sentence, stem=True)

df_token = pd.DataFrame(tokened_sentence, columns = ['word','class'])
df_token = df_token[(df_token['class']=='Noun') |
                    (df_token['class']=='Verb') |
                    (df_token['class']=='Adjective')]

df_stopwords = pd.read_csv('datasets/stopwords.csv')
stopwords = list(df_stopwords['stopword'])

keywords = []
for word in df_token.word:
    if len(word) > 1 and word not in stopwords:
        keywords.append(word)

embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')

sim_words = []
for keyword in keywords:
    try:
        sim_word = embedding_model.wv.most_similar(keyword, topn=10)
        for word, _ in sim_word:
            sim_words.append(word)
    except:
        continue
print(sim_words)
sentence = ' '.join(sim_words)
print(sentence)
sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation)
