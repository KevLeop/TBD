import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import NearestNeighbors

from evaluator import Evaluator
from dataset_handler import DatasetHandler

# Leer archivos
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

# Mostrar
print(ratings.head())
print(ratings.info())
print(users.head())
print(users.info())
print(movies.head())
print(movies.info())

movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].fillna("").astype('str')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]

titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def genre_recommendations(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]


print("*******************************************************************")

print (genre_recommendations('Good Will Hunting (1997)').head(5))
print("*******************************************************************")
print (genre_recommendations('Toy Story (1995)').head(2))
print("*******************************************************************")
print (genre_recommendations('Saving Private Ryan (1998)').head(10))

