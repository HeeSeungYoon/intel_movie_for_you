import pandas as pd
from gensim.models import Word2Vec

df_review = pd.read_csv('./preprocessing.csv')
df_review.info()

reviews = list(df_review['review'])

tokens = []
for review in reviews:
    token = review.split()
    tokens.append(token)

embedding_model = Word2Vec(tokens, vector_size=100, window=4, min_count=20, workers=12, epochs=100, sg=1)
embedding_model.save('./models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))