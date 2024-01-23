import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from imdb import Cinemagoer, IMDbDataAccessError
import streamlit as st
import time

links_df = pd.read_csv('links.csv')
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
tags_df = pd.read_csv('tags.csv')

ia = Cinemagoer()

def get_movie_poster(title, max_retries=3):
    retry_count = 0
    backoff_factor = 1  # seconds

    while retry_count < max_retries:
        try:
            search_results = ia.search_movie(title)
            if search_results:
                movie_id = search_results[0].movieID
                movie = ia.get_movie(movie_id)
                if 'cover url' in movie:
                    return movie['cover url']
            return None

        except IMDbDataAccessError as e:
            if 'timeout' in str(e).lower():
                print(f"Timeout occurred for {title}, retrying... (Attempt {retry_count + 1} of {max_retries})")
                time.sleep(backoff_factor * (2 ** retry_count))  # Exponential backoff
                retry_count += 1
            else:
                print(f"An IMDb data access error occurred for {title}: {e}")
                return None

        except Exception as e:
            print(f"An error occurred while fetching poster for {title}: {e}")
            return None

    print(f"Maximum retries reached for {title}. No poster available.")
    return None


# Filter by year
def filter_movies_by_year(movies_df, start_year, end_year):
    return movies_df[(movies_df['year'] >= start_year) & (movies_df['year'] <= end_year)]

# Popularity-Based Recommender
def get_top_n_movies(n, start_year, end_year):
    movie_ratings = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_ratings.columns = ['average_rating', 'rating_count']
    filtered_movies = movie_ratings[movie_ratings['rating_count'] >= 2]
    
    sorted_movies = filtered_movies.sort_values(by=['average_rating', 'rating_count'], ascending=[False, False])
    
    top_movies = sorted_movies.merge(movies_df, on='movieId').reset_index(drop=True)
    
    top_movies = filter_movies_by_year(top_movies, start_year, end_year)
    return top_movies[['title', 'year', 'genres']].head(n)



# Item-Based Collaborative Filtering
movie_user_matrix = ratings_df.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
movie_similarity = cosine_similarity(movie_user_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# def get_similar_movies(movie_id, n):
#     if movie_id not in movie_similarity_df.index:
#         raise ValueError("Movie ID not found in the database.")
#     similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]
#     similar_movies_df = movies_df[movies_df['movieId'].isin(similar_movies.index)].copy()
#     similar_movies_df['similarity'] = similar_movies.values

#     # Filter movies based on the year range
#     top_movies = similar_movies_df[(similar_movies_df['year'] >= start_year) & (similar_movies_df['year'] <= end_year)]
#     return top_movies.head(n)

def get_similar_movies_by_title(title, n):
    
    closest_title = process.extractOne(title, movies_df['title'].values)[0]
   
    movie_id = movies_df[movies_df['title'] == closest_title]['movieId'].iloc[0]
   
    similar_scores = movie_similarity_df.loc[movie_id]

    top_similar_scores = similar_scores.sort_values(ascending=False)[1:n+1]
    
    top_similar_movie_ids = top_similar_scores.index
    
    similar_movies = movies_df[movies_df['movieId'].isin(top_similar_movie_ids)]
    
    similar_movies = similar_movies.assign(similarity=similar_movies['movieId'].map(top_similar_scores))
    
    if start_year is not None and end_year is not None:
        similar_movies = filter_movies_by_year(similar_movies, start_year, end_year)
    return similar_movies.head(n)

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

    recommended_movies_df = movies_df[movies_df["movieId"].isin(recommended_movies.head(n).index)].copy()
    recommended_movies_df.loc[:, "estimated_rating"] = recommended_movies.head(n)

    return recommended_movies_df.head(n)

def get_movies_by_mood(mood, n):
    mood_movies = tags_df[tags_df['tag'].str.contains(mood, case=False, na=False)]
    
    # Merge with movies data to get movie details
    mood_movies_details = mood_movies.merge(movies_df, on='movieId').drop_duplicates('movieId')
    
    if start_year is not None and end_year is not None:
        mood_movies_details = filter_movies_by_year(mood_movies_details, start_year, end_year)
    return mood_movies_details.head(n)


st.set_page_config(page_title="Go WBSFLIX",page_icon="🍿",layout="wide")
# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://wallpaperaccess.com/full/3658597.jpg" alt="Banner Image">
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
        object-position: center;

    }
</style>
"""
st.title('GO WBS FLIX')
# Display the custom HTML
st.components.v1.html(custom_html)

# Sidebars for user input
st.sidebar.title('Get Recommendations')
option = st.sidebar.selectbox('Choose your recommendation type:', ['Top Movies', 'Similar Movies by Titles', 'User Recommendations','Tag-Based Movies'])

if option == 'Top Movies':
    st.title('Top Movies')
    start_year, end_year = st.sidebar.select_slider(
        'Select a range of years',
        options=list(range(1960, 2024)), 
        value=(1990, 2010)  
    )
    n = st.sidebar.slider('Number of top movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Top Movies'):
        top_movies = get_top_n_movies(n, start_year, end_year)
        movie_counter = 0

        for index, row in top_movies.iterrows():
            # Every 5 movies, start a new row
            if movie_counter % 2 == 0:
                cols = st.columns(2)  # Create 5 columns
                current_col_index = 0

            with cols[current_col_index]:
                st.markdown(f"### {row['title']}")
                st.markdown(f"**Year**: {int(row['year'])}")
                st.markdown(f"**Genres**: {row['genres']}")
                poster_url = get_movie_poster(row['title'])
                if poster_url:
                    st.image(poster_url, width=200)
                st.write("----")
            current_col_index = (current_col_index + 1) % 5
            movie_counter += 1

elif option == 'Similar Movies by Titles':
    st.title('Top Movies by Titles')
    title = st.sidebar.text_input('Enter Movie Title:')
    start_year, end_year = st.sidebar.select_slider(
        'Select a range of years',
        options=list(range(1960, 2024)), 
        value=(1990, 2010)  
    )
    n = st.sidebar.slider('Number of similar movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Similar Movies'):
        try:
            similar_movies = get_similar_movies_by_title(title, n)
            movie_counter = 0
            for index, row in similar_movies.iterrows():
                if movie_counter % 2 == 0:
                    cols = st.columns(2)  # Create 2 columns
                    current_col_index = 0

                with cols[current_col_index]:
                    st.markdown(f"### {row['title']}")
                    st.markdown(f"**Year**: {int(row['year'])}")
                    st.markdown(f"**Genres**: {row['genres']}")
                    poster_url = get_movie_poster(row['title'])
                    if poster_url:
                        st.image(poster_url, width=200)
                    st.write("----")

                current_col_index = (current_col_index + 1) % 2
                movie_counter += 1
        except ValueError as e:
            st.error(e)



elif option == 'User Recommendations':
    st.title('Top Movies by User')
    user_id = st.sidebar.number_input('Enter User ID:', min_value=1)
    start_year, end_year = st.sidebar.select_slider(
        'Select a range of years',
        options=list(range(1960, 2024)), 
        value=(1990, 2010)  
    )
    n = st.sidebar.slider('Number of movies to recommend:', 1, 20, 5)
    if st.sidebar.button('Show Recommendations'):
        try:
            user_recommendations = get_recommendations_for_user(user_id, n)
            movie_counter = 0
            for index, row in user_recommendations.iterrows():
                if movie_counter % 2 == 0:
                    cols = st.columns(2)
                    current_col_index = 0

                with cols[current_col_index]:
                    st.markdown(f"### {row['title']}")
                    st.markdown(f"**Year**: {int(row['year'])}")
                    st.markdown(f"**Genres**: {row['genres']}")
                    poster_url = get_movie_poster(row['title'])
                    if poster_url:
                        st.image(poster_url, width=200)
                    st.write("----")

                current_col_index = (current_col_index + 1) % 2
                movie_counter += 1
        except ValueError as e:
            st.error(e)

elif option == 'Tag-Based Movies':
    st.title('Top Movies by Tag')
    mood = st.sidebar.text_input('Enter Mood:')
    start_year, end_year = st.sidebar.select_slider(
        'Select a range of years',
        options=list(range(1960, 2024)), 
        value=(1990, 2010)  
    )
    n = st.sidebar.slider('Number of mood-based movies to display:', 1, 20, 5)
    if st.sidebar.button('Show Movies'):
        try:
            mood_movies = get_movies_by_mood(mood, n)
            movie_counter = 0
            for index, row in mood_movies.iterrows():
                if movie_counter % 2 == 0:
                    cols = st.columns(2)
                    current_col_index = 0

                with cols[current_col_index]:
                    st.markdown(f"### {row['title']}")
                    st.markdown(f"**Year**: {int(row['year'])}")
                    st.markdown(f"**Mood**: {row['tag']}")
                    poster_url = get_movie_poster(row['title'])
                    if poster_url:
                        st.image(poster_url, width=200)
                    st.write("----")

                current_col_index = (current_col_index + 1) % 2
                movie_counter += 1
        except ValueError as e:
            st.error(e)

