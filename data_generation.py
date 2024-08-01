import pandas as pd
import numpy as np
import random
import openai
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.quantization import quantize_dynamic
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

def collate_fn(batch):
    user_embeddings1, user_embeddings2, user_embeddings3, user_embeddings4, user_embeddings5 = [], [], [], [], []
    movie_embeddings, ratings = [], []

    for user_embs, movie_emb, rating in batch:
        user_embeddings1.append(user_embs[0])
        user_embeddings2.append(user_embs[1])
        user_embeddings3.append(user_embs[2])
        user_embeddings4.append(user_embs[3])
        user_embeddings5.append(user_embs[4])
        movie_embeddings.append(movie_emb)
        ratings.append(rating)

    user_embeddings1 = np.array(user_embeddings1)
    user_embeddings2 = np.array(user_embeddings2)
    user_embeddings3 = np.array(user_embeddings3)
    user_embeddings4 = np.array(user_embeddings4)
    user_embeddings5 = np.array(user_embeddings5)

    movie_embeddings = np.array(movie_embeddings)
    ratings = np.array(ratings)

    user_embeddings1 = torch.tensor(user_embeddings1, dtype=torch.float32)
    user_embeddings2 = torch.tensor(user_embeddings2, dtype=torch.float32)
    user_embeddings3 = torch.tensor(user_embeddings3, dtype=torch.float32)
    user_embeddings4 = torch.tensor(user_embeddings4, dtype=torch.float32)
    user_embeddings5 = torch.tensor(user_embeddings5, dtype=torch.float32)
    movie_embeddings = torch.tensor(movie_embeddings, dtype=torch.float32)
    ratings = torch.tensor(ratings, dtype=torch.float32)

    return (user_embeddings1, user_embeddings2, user_embeddings3, user_embeddings4, user_embeddings5), movie_embeddings, ratings


class DataGeneration(object):

    def __init__(self, OPENAI_API_KEY):
        openai.api_key = OPENAI_API_KEY

    def data_generator(self, movie_id_dict, embedding_dict, ratings_df, user_ids, batch_size):
        dataset = MovieLensDataset(movie_id_dict, embedding_dict, ratings_df, user_ids)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

    def get_embedding(self, text):
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        embedding = model([text])[0,:]
        return embedding

    def get_summary(self, movie_title, max_tokens=1000):
        system_dict = {"role": "system", "content": "You are a film expert who is tasked with providing a brief summary of a movie."}
        messages = [system_dict]
        user_dict = {"role": "user", "content": "Summarize the plot of the movie: " + movie_title}
        messages.append(user_dict)
        
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0125',
            messages=messages,
            max_tokens=max_tokens
        ).choices[0].message['content']

        return response

class MovieLensDataset(Dataset):
    def __init__(self, movie_id_dict, embedding_dict, ratings_df, user_ids):
        self.movie_id_dict = movie_id_dict
        self.embedding_dict = embedding_dict
        self.ratings_df = ratings_df
        self.user_ids = user_ids

        self.data = []
        for user_id in self.user_ids:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            for _, row in user_ratings.iterrows():
                movie_id = row['movie_id']
                rating = row['rating']
                try:
                    movie_title = self.movie_id_dict[movie_id]
                    movie_embedding = self.embedding_dict[movie_title]
                except KeyError:
                    continue

                if rating == 0:
                    continue

                one_hot = [0] * 5
                one_hot[rating - 1] = 1
                user_embeddings = self.get_user_history_embeddings(user_id)
                self.data.append((user_embeddings, movie_embedding, np.array(one_hot)))

    def get_user_history_embeddings(self, user_id):
        emb1, emb2, emb3, emb4, emb5 = [], [], [], [], []
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        for _, row in user_ratings.iterrows():
            try:
                movie_id = row['movie_id']
                rating = row['rating']
                movie_title = self.movie_id_dict[movie_id]
                embedding = self.embedding_dict[movie_title]
            except KeyError:
                continue

            if rating == 1:
                emb1.append(embedding)
            elif rating == 2:
                emb2.append(embedding)
            elif rating == 3:
                emb3.append(embedding)
            elif rating == 4:
                emb4.append(embedding)
            elif rating == 5:
                emb5.append(embedding)

        empty_embedding = np.zeros_like(list(self.embedding_dict.values())[0])
        emb1 = np.mean(emb1, axis=0) if emb1 else empty_embedding
        emb2 = np.mean(emb2, axis=0) if emb2 else empty_embedding
        emb3 = np.mean(emb3, axis=0) if emb3 else empty_embedding
        emb4 = np.mean(emb4, axis=0) if emb4 else empty_embedding
        emb5 = np.mean(emb5, axis=0) if emb5 else empty_embedding

        return emb1, emb2, emb3, emb4, emb5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_embeddings, movie_embedding, one_hot = self.data[idx]
        return user_embeddings, movie_embedding, one_hot
