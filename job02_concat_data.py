import pandas as pd
import glob

data_path = glob.glob('./crawling_data/*')

df = pd.DataFrame()
for path in data_path:
    print(path)
    df_review = pd.read_csv(path)
    df_review.dropna(inplace=True)
    df_review.drop_duplicates(inplace=True)
    df = pd.concat([df, df_review], ignore_index=True)

    # titles = set(df_review.index.to_list())
    # for title in titles:
    #     try:
    #         total_review = ''
    #         reviews = df_review.loc[title]
    #         reviews = reviews['review'].to_list()
    #         for review in reviews:
    #             if review is None:
    #                 continue
    #             total_review += review
    #         dict = {'title':[title],'review':[total_review]}
    #         df_movie = pd.DataFrame(dict)
    #         df = pd.concat([df, df_movie], ignore_index=True)
    #     except:
    #         continue

df.drop_duplicates(inplace=True)
df.info()

df.to_csv('./crawling_data/movie_reviews_201601_202310.csv',index=False)
