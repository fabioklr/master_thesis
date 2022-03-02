"""Preprocess Twitter data.
"""

import pandas as pd
import os
from preprocessor import tokenize
from multiprocessing import Pool
from time import time
from tweepy import OAuthHandler, API
from torch import save, load
from deepl import Translator
from re import findall
from ekphrasis.classes.segmenter import Segmenter
from keys import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import pyarrow.ipc as ipc
# import pyarrow.feather as feather
# import pyarrow as pa
# import pandas as pd
# import numpy as np

# Set pandas options.
pd.set_option('display.min_rows', 400)
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# def read_feather_in_chunks(filepath):
#     with ipc.RecordBatchFileReader(filepath) as reader:
#         for batch_index in range(reader.num_record_batches):
#             batch = reader.get_batch(batch_index)
#             print(f'Read in batch {batch_index} which had {batch.num_rows} rows')
#             data_df = batch.to_pandas(use_threads=True, timestamp_as_object=True, )
#             yield data_df

# Clean text, and find, split and replace hashtags.
seg_tw = Segmenter(corpus = "twitter")
def clean(tweet_text):
    hashtags = findall(r"#(\w+)", tweet_text)
    tweet_text = tokenize(tweet_text)
    if hashtags:
        hashtags = [seg_tw.segment(hashtag) for hashtag in hashtags]
        while hashtags:
            tweet_text = tweet_text.replace('$HASHTAG$', hashtags.pop(0), 1)
    return tweet_text

def main():
    startTime = time()

    # Import tweets and remove those whose language is undefined or not translatable
    tweets = pd.read_feather('data/organic_tweets.feather')
    # tweets = tweets.sample(n = 400)

    if 'processed_text' not in tweets.columns:
        del tweets['Unnamed: 0']

        # Filter languages with less than 100 tweets
        langs_less_than_hundred = [lang for lang, size in tweets.groupby('lang').size().iteritems() if size < 100]
        tweets = tweets[-tweets['lang'].isin(langs_less_than_hundred)]

        # Filter languages that are undefined, non-translatable or with a non-Latin alphabet
        tweets = tweets[-tweets['lang'].isin(['und', 'iw', 'ja', 'ar', 'zh', 'th', 'ko', 'el'])]

        # Filter authors with less than 100 tweets
        authors_less_than_hundred = [id for id, size in tweets.groupby('author_id').size().iteritems() if size < 100]
        tweets = tweets[-tweets['author_id'].isin(authors_less_than_hundred)]
    
        pool = Pool()
        result = pool.map(clean, tweets.text.values.tolist())
        pool.close()
        pool.join()

        tweets['processed_text'] = result

        # Remove duplicate rows, i.e. keep only one of multiple tweets with identical author_id and text, and reset the index to 0, 1, ..., n-1
        tweets = tweets.drop_duplicates(subset = ['author_id', 'processed_text'], ignore_index = True)
    
    # Remove tweets that are too long for Helsinki University translation models and reset the index.
    tweets = tweets[-(tweets.processed_text.str.len() > 512)]
    tweets = tweets.reset_index(drop = True)
        
    # Get and add Twitter screen handles - the modifiable Twitter "name" seen by users -
    # to corresponding rows
    if 'screen_handle' not in tweets.columns:
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = API(auth)

        twitter_ids = set(tweets.author_id)
        id_to_handle = {}

        for id in twitter_ids:
            print(id)
            handle = api.get_user(user_id = id).screen_name
            id_to_handle[id] = handle

        for row in tweets.itertuples():
            tweets.at[row.Index, 'screen_handle'] = id_to_handle[row.author_id]

    # Translate raw tweets
    # Find out translation cost by calculating total number of characters in non-English tweets
    tweet_lengths = [len(element) for element in tweets.loc[tweets['lang'] != 'en'].text.tolist()]
    total_n_char = float(sum(tweet_lengths))
    cost_million_chars = 20
    print("As of December 2021, translating all non-English tweets with the DeepL API would cost EUR", total_n_char / 1000000 * cost_million_chars - 10)

    # Translate Roman and other languages with Helsinki models, and the rest with DeepL.
    roman_langs = ['pt', 'fr', 'it', 'es', 'ca']
    helsinki_langs = ['de', 'sv', 'da', 'tr', 'fi', 'pl', 'et', 'nl', 'vi']
    other_langs = [lang for lang in set(tweets.lang) if lang not in roman_langs and lang not in helsinki_langs and lang != 'en']
    lang_groups = tweets.groupby('lang')
    translator = Translator(authentication_key) # Initiate DeepL.
    if 'translated' not in tweets.columns:
        tweets['translated'] = False

    try:
        os.makedirs(f'{(cwd := os.getcwd())}/models/Helsinki-NLP')
    except Exception:
        pass

    def translate():
        for lang, group in lang_groups:
            print(lang)
            if lang in roman_langs:
                if os.path.isfile(f'{cwd}/models/Helsinki-NLP/opus-mt-ROMANCE-en.pt'):
                    roman_tokenizer = load(f'{cwd}/models/Helsinki-NLP/opus-mt-ROMANCE-en_tokenizer.pt')
                    roman_model = load(f'{cwd}/models/Helsinki-NLP/opus-mt-ROMANCE-en.pt')
                else:
                    roman_tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')
                    save(roman_tokenizer, f'{cwd}/models/Helsinki-NLP/opus-mt-ROMANCE-en_tokenizer.pt')
                    roman_model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')
                    save(roman_model, f'{cwd}/models/Helsinki-NLP/opus-mt-ROMANCE-en.pt')
                for row in group.itertuples():
                    if not group.at[row.Index, 'translated']:
                        batch = roman_tokenizer([row.processed_text], return_tensors="pt", padding=True)
                        translation = roman_model.generate(**batch)
                        group.at[row.Index, 'processed_text'] = roman_tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
            elif lang in helsinki_langs:
                model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
                if os.path.isfile(f'{cwd}/models/{model_name}.pt'):
                    tokenizer = load(f'{cwd}/models/{model_name}_tokenizer.pt')
                    model = load(f'{cwd}/models/{model_name}.pt')
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    save(tokenizer, f'{cwd}/models/{model_name}_tokenizer.pt')
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    save(model, f'{cwd}/models/{model_name}.pt')
                for row in group.itertuples():
                    if not group.at[row.Index, 'translated']:
                        batch = tokenizer([row.processed_text], return_tensors="pt", padding=True)
                        translation = model.generate(**batch)
                        group.at[row.Index, 'processed_text'] = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
            elif lang in other_langs:
                for row in group.itertuples():
                    if not group.at[row.Index, 'translated']:
                        group.at[row.Index, 'processed_text'] = translator.translate_text(row.processed_text, target_lang = "EN-GB").text
            
            yield group

    generator = translate()
    for group in generator:
        for row in group.itertuples():
            if not tweets.at[row.Index, 'translated']:
                tweets.at[row.Index, 'processed_text'] = row.processed_text
                tweets.at[row.Index, 'translated'] = True
    
    for row in tweets.itertuples():
        editing = tweets.at[row.Index, 'processed_text'].split()
        new_processed = [word for word in editing if '$' not in word]
        tweets.at[row.Index, 'processed_text'] = ' '.join(new_processed)

    tweets.to_feather('data/preprocessed_tweets.feather')
    
    print(len(tweets))
    executionTime = time() - startTime
    print(f'Execution time in seconds: {executionTime}')

if __name__ == '__main__':
    main()