"""Clustering.
"""

import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from hdbscan import HDBSCAN
from time import time
from umap import UMAP
from torch import save, load
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from resource import getrusage, RUSAGE_SELF

def compute_sentence_embeddings(tweets, model_name):
    if os.path.isfile(f'{(cwd := os.getcwd())}/models/{model_name}.pt'):
        model = load(f'{cwd}/models/{model_name}.pt')
    else:
        model = SentenceTransformer(model_name)
        os.mkdir(f'{cwd}/models')
        save(model, f'{cwd}/models/{model_name}.pt')

    embeddings = model.encode(tweets.processed_text.values, show_progress_bar = True)

    #Store sentences & embeddings on disc
    with open('data/embeddings.pkl', "wb") as fOut:
        pickle.dump({'embeddings': embeddings}, fOut, protocol = pickle.HIGHEST_PROTOCOL)
    
    return embeddings

def c_tf_idf(documents, m, ngram_range = (1, 1)):
    count = CountVectorizer(ngram_range = ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis = 1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis = 0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n = 20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.topic) # changed '.Topic'
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['topic'])
                    .processed_text
                    .count()
                    .reset_index()
                    .rename({"topic": "Topic", "processed_text": "Size"}, axis='columns')
                    .sort_values("Size", ascending=False))
    
    return topic_sizes

def main():
    startTime = time()
    tweets = pd.read_feather('data/preprocessed_tweets.feather')
    del tweets['translated']
    # tweets = tweets.sample(frac = 0.2, random_state = 123)
    # tweets_test = tweets.drop(tweets.index).sample(frac = 1, random_state = 124)
    # del tweets

    if os.path.isfile(f'{(cwd := os.getcwd())}/data/embeddings.pkl'):
        #Load sentences & embeddings from disc
        with open('data/embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            embeddings = stored_data['embeddings']
    else:
        embeddings = compute_sentence_embeddings(tweets = tweets, model_name = 'all-MiniLM-L6-v2')

    if os.path.isfile(f'{cwd}/data/umap_embeddings.pkl'):
        with open('data/umap_embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            umap_embeddings = stored_data['umap_embeddings']  
    else:
        # Reduce dimensionality.
        dimensionality_reducer = UMAP(n_components = 10, n_neighbors = 100, min_dist = 0.01, verbose = True, low_memory = True)
        umap_embeddings = dimensionality_reducer.fit_transform(embeddings)
        with open('data/umap_embeddings.pkl', 'wb') as fOut:
            pickle.dump({'umap_embeddings': umap_embeddings}, fOut, protocol = pickle.HIGHEST_PROTOCOL)

    # Cluster vectors.
    cluster_model = HDBSCAN(min_cluster_size = 7000, min_samples = 30, cluster_selection_method = 'leaf', core_dist_n_jobs = 1)
    cluster = cluster_model.fit(umap_embeddings)
    tweets['topic'] = cluster.labels_

    docs_per_topic = tweets.groupby(['topic'], as_index = False).agg({'processed_text': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.processed_text.values, m = len(tweets))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n = 10)
    w = csv.writer(open('data/top_n_words_per_topic.csv', 'w'))
    for key, val in top_n_words.items():
        w.writerow([key, val])

    tweets['topic_top_words'] = ''
    tweets['topic_top_words'] = tweets['topic_top_words'].astype(object)
    for row in tweets.itertuples():
        tweets.at[row.Index, 'topic_top_words'] = [top_n_words[row.topic][i][0] for i in range(5)]

    tweets = tweets.reset_index()
    tweets.to_feather('data/final_tweets.feather')

    print(f'Maximum memory used: {getrusage(RUSAGE_SELF).ru_maxrss}')
    print(f'Execution time in seconds: {time() - startTime}')

main()