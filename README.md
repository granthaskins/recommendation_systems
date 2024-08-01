# Movie Recommendation System

This project implements a movie recommendation system using various machine learning and deep learning libraries including PyTorch, TensorFlow, and Sentence Transformers. The system generates embeddings for user history and movie titles and utilizes these embeddings to predict movie ratings using the public 100k movies dataset. Available at: https://grouplens.org/datasets/movielens/100k/

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

    If `requirements.txt` is not provided, install the dependencies manually:

    ```bash
    pip install pandas numpy torch sentence-transformers tensorflow tensorflow-hub tqdm
    pip install openai transformers
    ```

## Usage

1. **Set up your OpenAI API key:**

    Ensure you have an OpenAI API key and set it as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-api-key'  # On Windows use `set OPENAI_API_KEY=your-api-key`
    ```

2. **Run the main script:**

    Run the provided Python scripts to generate the datasets and train the model.

    ```bash
    python get_dfs.py
    python recommendation_dl.py
    ```

    Make sure to customize the scripts as per your dataset paths and requirements.

## File Descriptions

- **get_dfs.py:** This script processes the raw datasets and prepares the necessary DataFrames for the recommendation system.
- **recommendation_dl.py:** The main script that contains the classes and functions for generating embeddings, preparing data loaders, and training the recommendation model.

## Dependencies

- `pandas`
- `numpy`
- `torch`
- `sentence-transformers`
- `tensorflow`
- `tensorflow-hub`
- `tqdm`
- `openai`
- `transformers`

Install these dependencies using the command:

```bash
pip install pandas numpy torch sentence-transformers tensorflow tensorflow-hub tqdm openai transformers
