import pandas as pd
from konlpy.tag import Okt
import re
from tqdm import tqdm

df = pd.read_csv('./crawling_data/movie_reviews_201601_202310.csv')
df.info()

okt = Okt()

df_stopwords = pd.read_csv('./datasets/stopwords.csv')
stopwords = list(df_stopwords['stopword'])

reviews = []
for review in tqdm(df.review, desc='Preprocessing review', mininterval=0.01):
    review = re.sub('[^가-힣]',' ',review)
    tokened_review = okt.pos(review, stem=True)

    df_token = pd.DataFrame(tokened_review, columns=['word','class'])
    df_token = df_token[(df_token['class']=='Noun') |
                        (df_token['class']=='Verb') |
                        (df_token['class']== 'Adjective')]

    words = []
    for word in df_token.word:
        if 1 < len(word):
            if word not in stopwords:
                words.append(word)
    cleaned_review = ' '.join(words)
    reviews.append(cleaned_review)

df['review'] = reviews
df.info()
df.to_csv('./preprocessing.csv',index=False)

