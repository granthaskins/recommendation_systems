import pandas as pd
import argparse

def load_dfs(ratings_df_path="./ml-100k/u.data",users_df_path="./ml-100k/u.user",
                    items_df_path="./ml-100k/u.item"):

    columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    print(f"Loading dataset from {ratings_df_path}...")
    try:
        ratings_df = pd.read_csv(ratings_df_path, sep='\t', names=columns, engine='python')
        print("Ratings dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    try:
        users_df = pd.read_csv(users_df_path, encoding='latin1')
    except Exception as e:
        print(f"Error loading users dataset: {e}")
        exit()

    items_columns = [
        'movie_id', 'movie_title', 'release_date', 'video_release_date',
        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
        "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]

    # Read the items dataset using '|' as the delimiter
    try:
        items_df = pd.read_csv(items_df_path, sep='|', names=items_columns, encoding='latin1')
    except Exception as e:
        print(f"Error loading items dataset: {e}")
        exit()

    return ratings_df, users_df, items_df

def get_movie_dict(ratings_df_path="./ml-100k/u.data", users_df_path="./ml-100k/u.user",
                    items_df_path="./ml-100k/u.item"):

    _, _, items_df = load_dfs(ratings_df_path, users_df_path, items_df_path)

    # Create the dictionary with movie_id as keys and movie_title as values
    return items_df.set_index('movie_id')['movie_title'].to_dict()






