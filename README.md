# Movie Recommendation System

This macos project implements a movie recommendation system using various machine learning and deep learning libraries including PyTorch, TensorFlow, and Sentence Transformers (note just change device from 'mps' to 'cuda' if you are using Windows/Linux). The system generates embeddings for user history and movie titles and utilizes these embeddings to predict movie ratings using the public 100k movies dataset. Available at: https://grouplens.org/datasets/movielens/100k/

In short, this method uses cheap gpt queries to generate summaries from movie titles alone that are then mapped to text embedding to allow for a deep learning based recommendation system solution. This framework can readily be adapted to other domains.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- Generates embeddings for movie titles using the Universal Sentence Encoder from TensorFlow Hub.
- Predicts movie ratings based on user history using PyTorch.
- Utilizes OpenAI's GPT-3.5 for generating movie summaries.
- Implements a custom DataLoader and Dataset for efficient data handling and batching.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/movierecommendation.git
    cd movierecommendation
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

**Run the main script:**

    Run the provided Python scripts to generate the datasets and train the model.

    ```bash
    python get_dfs.py
    python recommendation_dl.py
    ```

    Make sure to customize the scripts as per your dataset paths and requirements.

## File Descriptions

- **get_dfs.py:** This script processes the raw datasets and prepares the necessary DataFrames for the recommendation system.
- **recommendation_dl.py:** The main script that contains the classes and functions for generating embeddings, preparing data loaders, and training the recommendation model.
- **data_generation.py:** This script creates the dataset and the dataloaders using gpt to create the text summaries that are used to generate embeddings.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
