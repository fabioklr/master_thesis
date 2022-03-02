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

def main():
    startTime = time()

    tweets = pd.read_feather('data/preprocessed_tweets.feather')
    tweets = tweets.sample(n = int(0.25 * len(tweets)))

    if os.path.isfile(f'{(cwd := os.getcwd())}/data/embeddings.pkl'):
        #Load sentences & embeddings from disc
        with open('data/embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            embeddings = stored_data['embeddings']
    else:
        # Compute sentence embeddings
        model_name = 'all-MiniLM-L6-v2'
        if os.path.isfile(f'{cwd}/models/{model_name}.pt'):
            model = load(f'{cwd}/models/{model_name}.pt')
        else:
            model = SentenceTransformer(model_name)
            os.mkdir(f'{cwd}/models')
            save(model, f'{cwd}/models/{model_name}.pt')
    
        embeddings = model.encode(tweets.processed_text.values, show_progress_bar = True)

        #Store sentences & embeddings on disc
        with open('data/embeddings.pkl', "wb") as fOut:
            pickle.dump({'embeddings': embeddings}, fOut, protocol = pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(f'{cwd}/data/umap_embeddings.pkl'):
        with open('data/umap_embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            umap_embeddings = stored_data['umap_embeddings']
            print('UMAP done.')
    else:
        # Reduce dimensionality.
        print(getrusage(RUSAGE_SELF).ru_maxrss)
        umap_embeddings = UMAP(n_components = 10, n_neighbors = 50, min_dist = 0.01, verbose = True, low_memory = True).fit_transform(embeddings)
        print('UMAP done.')

        with open('data/umap_embeddings.pkl', 'wb') as fOut:
            pickle.dump({'embeddings': embeddings, 'umap_embeddings': umap_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    cluster_model = HDBSCAN(min_cluster_size = 200, min_samples = 30, cluster_selection_method = 'leaf', core_dist_n_jobs = 1)
    cluster = cluster_model.fit(umap_embeddings)

    # docs_df = pd.DataFrame(tweets.processed_text.values, columns = ['Doc'])
    # docs_df['Topic'] = cluster.labels_
    # docs_df['Doc_ID'] = range(len(docs_df))
    # docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    tweets['topic'] = cluster.labels_
    docs_per_topic = tweets.groupby(['topic'], as_index = False).agg({'processed_text': ' '.join})

    def c_tf_idf(documents, m, ngram_range = (1, 1)):
        count = CountVectorizer(ngram_range = ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis = 1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis = 0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count
    
    tf_idf, count = c_tf_idf(docs_per_topic.processed_text.values, m = len(tweets)) # here .Doc.values was changed

    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
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

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n = 20)
    w = csv.writer(open("data/top_n_words_per_topic.csv", "w"))
    for key, val in top_n_words.items():
        w.writerow([key, val])

    # topic_sizes = extract_topic_sizes(tweets); topic_sizes.head(10)
    # for i in range(len(top_n_words)):
    #     print(top_n_words[i][:5])
    
    tweets['topic_top_words'] = ''
    tweets['topic_top_words'] = tweets['topic_top_words'].astype(object)
    for row in tweets.itertuples():
        tweets.at[row.Index, 'topic_top_words'] = [top_n_words[row.topic][i][0] for i in range(5)]

    tweets = tweets.reset_index()
    tweets.to_feather('data/final_tweets.feather')

    executionTime = time() - startTime
    print(f'Execution time in seconds: {executionTime}')

    print(getrusage(RUSAGE_SELF).ru_maxrss)

main()