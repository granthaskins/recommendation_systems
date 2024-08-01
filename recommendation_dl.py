import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
from get_dfs import load_dfs, get_movie_dict
from data_generation import DataGeneration, MovieLensDataset, collate_fn

import sys
import threading
# Importing 'sys' and 'threading' libraries for potential use in parallelization and system-specific parameters.
# These libraries are not currently used in the script, but are available for future contributors who may
# wish to implement parallelization or other system-related functionalities.

class NCF(nn.Module):
    def __init__(self, embedding_dim, embedding_size=200):
        super(NCF, self).__init__()

        self.user_embedding1 = nn.Embedding(embedding_dim, embedding_size)
        self.user_embedding2 = nn.Embedding(embedding_dim, embedding_size)
        self.user_embedding3 = nn.Embedding(embedding_dim, embedding_size)
        self.user_embedding4 = nn.Embedding(embedding_dim, embedding_size)
        self.user_embedding5 = nn.Embedding(embedding_dim, embedding_size)
        
        self.movie_embedding = nn.Embedding(embedding_dim, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, user_embedding1, 
                        user_embedding2, user_embedding3, 
                        user_embedding4, user_embedding5, movie_embedding):
        user_embedded1 = self.user_embedding1(user_embedding1)
        user_embedded2 = self.user_embedding2(user_embedding2)
        user_embedded3 = self.user_embedding2(user_embedding3)
        user_embedded4 = self.user_embedding2(user_embedding4)
        user_embedded5 = self.user_embedding2(user_embedding5)
        movie_embedded = self.movie_embedding(movie_embedding)

        x = torch.cat([user_embedded1, user_embedded2, user_embedded3, user_embedded4, user_embedded5, movie_embedded], dim=1)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

def kl_divergence(p, q):

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    # Ensure the distributions are normalized (i.e., sum to 1)
    p /= p.sum()
    q /= q.sum()
    
    # Compute the KL divergence
    kl_div = np.sum(rel_entr(p, q))
    return kl_div

def k_keys_closest_to_zero(d, k):

    # Sort the dictionary items based on the absolute value of their values
    sorted_items = sorted(d.items(), key=lambda item: abs(item[1]))

    # Select the top k items
    closest_k_items = sorted_items[:k]

    # Extract and return the keys of these items
    return [item[0] for item in closest_k_items]

def train_model(args, model, train_loader, val_loader, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc='Epoch {}'.format(epoch+1)):
            
            user_embeddings, movie_embedding, rating = batch
            
            user_embedding1 = user_embeddings[0].long().to(device)
            user_embedding2 = user_embeddings[1].long().to(device)
            user_embedding3 = user_embeddings[2].long().to(device)
            user_embedding4 = user_embeddings[3].long().to(device)
            user_embedding5 = user_embeddings[4].long().to(device)

            movie_embedding = movie_embedding.long().to(device)
            rating = rating.long().to(device)

            optimizer.zero_grad()
            output = model(user_embedding1, user_embedding2, user_embedding3, user_embedding4, user_embedding5, movie_embedding).squeeze()
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        eval(args, model, val_loader, device)

    return model

def eval(args, model, data_loader, device):

    model.eval()
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            user_embedding1, user_embedding2, user_embedding3, user_embedding4, user_embedding5, movie_embedding, rating = batch

            user_embedding1 = user_embedding1.long().to(device)
            user_embedding2 = user_embedding2.long().to(device)
            user_embedding3 = user_embedding3.long().to(device)
            user_embedding4 = user_embedding4.long().to(device)
            user_embedding5 = user_embedding5.long().to(device)
            movie_embedding = movie_embedding.long().to(device)
            rating = rating.long().to(device)

            output = model(user_embedding1, user_embedding2, user_embedding3, user_embedding4, user_embedding5, movie_embedding).squeeze()
            loss = criterion(output, rating)
            running_loss += loss.item()

    print(f"Validation Loss: {running_loss/len(data_loader):.4f}")

def recommend_topk(model, movie_embeddings_dict, user_id, ratings_df, movie_id_dict, k=5, device='cpu'):

    user_embedding1, user_embedding2, user_embedding3, user_embedding4, user_embedding5 = get_user_history_embeddings(user_id, ratings_df, movie_id_dict)

    movie_divs = {}

    for movie_title in movie_embeddings_dict.keys():

        movie_embedding = movie_embeddings_dict[movie_title]

        output = model(user_embedding1, user_embedding2, user_embedding3, user_embedding4, user_embedding5, movie_embedding)
        kl_div = kl_divergence(output, np.array([0,0,0,0,1]))

        movie_divs[movie_title] = kl_div

    return k_keys_closest_to_zero(movie_divs, k)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Deep Learning Based Recommendation')
    parser.add_argument('--rating_df_path', type=str, required=True, help='Path to the rating dataset')
    parser.add_argument('--user_df_path', type=str, required=True, help='Path to the user dataset')
    parser.add_argument('--item_df_path', type=str, required=True, help='Path to the item dataset')
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_recommendations', type=int, default=5)
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    device = torch.device("mps" if args.use_gpu and torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    datageneration = DataGeneration(args.OPENAI_API_KEY)

    ratings_df, users_df, items_df = load_dfs(args.rating_df_path, args.user_df_path, args.item_df_path)
    
    ratings_df = ratings_df.fillna(0)
    users_df = items_df.fillna('N/A')
    items_df = items_df.fillna('N/A')

    num_users = ratings_df['user_id'].nunique()
    num_movies = ratings_df['movie_id'].nunique()

    unique_users = ratings_df['user_id'].unique()
    unique_movie_ids = ratings_df['movie_id'].unique()
    np.random.shuffle(unique_users)

    train_size = int((1-args.val_size-args.test_size)*num_users)
    val_size = int(args.val_size*num_users)
    test_size = int(args.test_size*num_users)

    train_user_ids = unique_users[:train_size]
    val_user_ids = unique_users[train_size:train_size+val_size]
    test_user_ids = unique_users[train_size+val_size:]

    num_users_train = len(train_user_ids)
    num_users_val = len(val_user_ids)

    ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes.values
    ratings_df['movie_id'] = ratings_df['movie_id'].astype('category').cat.codes.values

    movie_id_dict = get_movie_dict()

    embedding_dict = {}

    count = 0

    for movie_title in tqdm(movie_id_dict.items(),desc='Generating movie summary embeddings'):

        if count == 1200:
            break

        summary = datageneration.get_summary(movie_title[1])
        embedding = datageneration.get_embedding(summary)
        embedding_dict[movie_title[1]] = embedding
        if embedding.shape != (512,):
            print(embedding.shape)
            assert False

        count += 1

    embedding_dim = embedding_dict[list(embedding_dict.keys())[0]].shape[0]

    train_dataset = MovieLensDataset(movie_id_dict, embedding_dict, ratings_df, train_user_ids)
    val_dataset = MovieLensDataset(movie_id_dict, embedding_dict, ratings_df, val_user_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = NCF(embedding_dim).to(device)
    print('Model Instantiated')

    model = train_model(args, model, train_loader, val_loader, device)

    recommendations = recommend_topk(model, embedding_dict, test_user_ids[0], ratings_df, movie_id_dict, k=args.num_recommendations, device=device)
    print(f"Top {args.num_recommendations} movie recommendations for user {test_user_ids[0]}: {recommendations}")
