"""Analysis.
"""

import pandas as pd
import numpy as np
import umap.plot

# UMAP fire graph
topics = pd.read_feather('data/final_tweets.feather', ['topic']).astype(np.float32)
with open('data/umap_embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            umap_embeddings = stored_data['umap_embeddings']

umap.plot.points(umap_embeddings, labels = topics, theme = 'fire')

# Load Forbes Global 2000 data
companies = pd.read_csv("data/Forbes Global 2000 - 2019.csv")
companies = companies[companies['Continent'] == 'Europe']
print((companies.iloc[:5]).to_markdown())