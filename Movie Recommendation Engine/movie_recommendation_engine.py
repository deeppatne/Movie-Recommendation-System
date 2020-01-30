import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Reading csv file
df = pd.read_csv("movie_dataset.csv")

# For printing  first 3  rows
df.head(3)

#for finding number of rows and columns
df.shape

# Create an extra column containing only required features
features = ['keywords','cast','genres','director']

df[features].head(3)

#Clean and preprocess the data
for feature in features:
    df[feature] = df[feature].fillna('') #Filling missing values
   # print(df[feature])

#A function to combine required features into a string
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

#Applying function to each row in the dataset
df["combined_features"] = df.apply(combine_features,axis=1)
#df["combined_features"]

#Print to see the new column
df.head(3)

# countvectorizer
count_matrix = CountVectorizer().fit_transform(df["combined_features"])

#print(count_matrix.toarray())

# cosine similarity from count matrix(cos theta)
cosine_sim = cosine_similarity(count_matrix)

print(cosine_sim)

#Number of rows and columns
cosine_sim.shape

#Title from the index
def get_title_from_index(index):
  return df[df.index == index]["title"].values[0]

#Index from the title
def get_index_from_title(title):
  return df[df.title == title]["index"].values[0]

# Movie which user likes
movie_user_likes = "The Amazing Spider-Man"

# Getting the movie index from the title
movie_index = get_index_from_title(movie_user_likes)

#Access the row, through the movies index, corresponding to this movie (the liked movie) in the similarity matrix, 
# by doing this we will get the similarity scores of all other movies from the current movie

#Enumerate means making a tuple of movie index and similarity scores
#  Before: row of similarity scores like this  [5 0.6 0.3 0.9] 
#  After :                                     [(0, 5) (1, 0.6) (2, 0.3) (3, 0.9)]
#  Its in the  form                             (movie index, similarity score)
similar_movies =  list(enumerate(cosine_sim[movie_index]))



#Sorting the list in descending order as we want the movies with highest similarity scores first
# A lso discarding the first element as its the  same movie
# x[1] = second element of tupple
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

# print(sorted_similar_movies)

# For printing first 50 similar movies

i=0
print("Top 50 similar movies to "+movie_user_likes+" are:")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]) )
    i=i+1
    if i>=50:
        break