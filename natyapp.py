import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import re


links_df = pd.read_csv('links.csv')
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
tags_df = pd.read_csv('tags.csv')

# Popularity-Based Recommender
def get_top_n_movies(n):
    movie_ratings = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_ratings.columns = ['average_rating', 'rating_count']
    filtered_movies = movie_ratings[movie_ratings['rating_count'] > 2]
    
    # Sorting by average_rating first, then by rating_count
    sorted_movies = filtered_movies.sort_values(by=['average_rating', 'rating_count'], ascending=[False, False])
    
    top_movies = sorted_movies.merge(movies_df, on='movieId').reset_index(drop=True)
    return top_movies['title'].head(n)


# Item-Based Collaborative Filtering
movie_user_matrix = ratings_df.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
movie_similarity = cosine_similarity(movie_user_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

def get_similar_movies(movie_id, n):
    if movie_id not in movie_similarity_df.index:
        raise ValueError("Movie ID not found in the database.")
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]
    similar_movies_df = movies_df[movies_df['movieId'].isin(similar_movies.index)].copy()
    similar_movies_df['similarity'] = similar_movies.values
    return similar_movies_df['title']

# User-Based Collaborative Filtering
user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_recommendations_for_user(user_id, n):

    if user_id not in user_similarity_df.index:
        raise ValueError("User not found")

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]

    similar_users_ratings = user_movie_matrix.loc[similar_users.index, :]

    recommended_movies = similar_users_ratings.mean(axis=0)

    already_rated = user_movie_matrix.loc[user_id] > 0

    recommended_movies = recommended_movies[~recommended_movies.index.isin(already_rated[already_rated].index)]

    recommended_movies_df = movies_df[movies_df["movieId"].isin(recommended_movies.head(n).index)]
    recommended_movies_df["estimated_rating"] = recommended_movies.head(n).values

    return recommended_movies_df

def get_similar_movies_by_title(title, n):
    # Find the closest match to the given title
    closest_title = process.extractOne(title, movies_df['title'].values)[0]

    # Get the movieId for the closest title
    movie_id = movies_df[movies_df['title'] == closest_title]['movieId'].iloc[0]

    # Get similarity scores for the movieId
    similar_scores = movie_similarity_df.loc[movie_id]

    # Sort the scores in descending order and select top n scores
    top_similar_scores = similar_scores.sort_values(ascending=False)[1:n+1]

    # Get movie IDs for the top similar movies
    top_similar_movie_ids = top_similar_scores.index

    # Retrieve movie titles for these IDs
    similar_movies = movies_df[movies_df['movieId'].isin(top_similar_movie_ids)]

    # Add similarity scores
    similar_movies['similarity'] = similar_movies['movieId'].apply(lambda x: top_similar_scores[x])

    # Return only the titles and similarity scores
    return similar_movies[['title', 'similarity']]

def get_movies_by_mood(mood, n):
    # Filter tags data for the given mood
    mood_movies = tags_df[tags_df['tag'].str.contains(mood, case=False, na=False)]
    
    # Merge with movies data to get movie details
    mood_movies_details = mood_movies.merge(movies_df, on='movieId').drop_duplicates('movieId')
    
    return mood_movies_details.head(n)

def get_movies_by_year(year, n):
    
    movie_ratings = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_ratings.columns = ['average_rating', 'rating_count']
    
    filtered_movies = movie_ratings[movie_ratings['rating_count'] > 2]

    movies_with_ratings = movies_df.merge(filtered_movies, left_on='movieId', right_on='movieId', how='left')

    year_movies = movies_with_ratings[movies_with_ratings['year'] == year].sort_values(by='average_rating', ascending=False)

    return year_movies.head(n)[['title', 'genres']].values.tolist()

st.image('https://wallpaperaccess.com/full/3658597.jpg')

# Title of your app
st.title('Movie Recommendation System')

# Sidebars for user input
st.sidebar.title('Get Recommendations')
option = st.sidebar.selectbox('Choose your recommendation type:', ['Top Movies', 'Similar Movies by Titles','Similar Movies by Id', 'User Recommendations','Mood-Based Movies','Movies by Year'])

if option == 'Top Movies':
    n = st.sidebar.slider('Number of top movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Top Movies'):
        top_movies = get_top_n_movies(n)
        st.write(top_movies)

elif option == 'Similar Movies by Id':
    movie_id = st.sidebar.number_input('Enter Movie ID:', min_value=1)
    n = st.sidebar.slider('Number of similar movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Similar Movies'):
        try:
            similar_movies = get_similar_movies(movie_id, n)
            st.write(similar_movies)
        except ValueError as e:
            st.error(e)

elif option == 'Similar Movies by Titles':
    title = st.sidebar.text_input('Enter Movie Title:')
    n = st.sidebar.slider('Number of similar movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Similar Movies'):
        try:
            similar_movies = get_similar_movies_by_title(title, n)
            st.write(similar_movies['title'])
        except ValueError as e:
            st.error(e)

elif option == 'User Recommendations':
    user_id = st.sidebar.number_input('Enter User ID:', min_value=1)
    n = st.sidebar.slider('Number of movies to recommend:', 1, 20, 5)
    if st.sidebar.button('Show Recommendations'):
        try:
            user_recommendations = get_recommendations_for_user(user_id, n)
            st.write(user_recommendations)
        except ValueError as e:
            st.error(e)

elif option == 'Mood-Based Movies':
    mood = st.sidebar.text_input('Enter Mood:')
    n = st.sidebar.slider('Number of mood-based movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Movies'):
        mood_movies = get_movies_by_mood(mood, n)
        st.write(mood_movies)

elif option == 'Movies by Year':
    year = st.sidebar.number_input('Enter Year:', min_value=1900, max_value=2024, value=1999)
    n = st.sidebar.slider('Number of movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Movies'):
        year_movies_info = get_movies_by_year(year, n)
    for title, genres in year_movies_info:
        st.write(f"{title} - Genres: {genres}")

