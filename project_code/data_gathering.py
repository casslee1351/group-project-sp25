import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

import numpy as np
import pandas as pd
from collections import Counter

import dask.dataframe as dd

def remove_between_brackets(text):
  """Removes all text between any matching pair of brackets, including the brackets themselves."""
  return re.sub(r'\[.*?\]', '', text)

def cleanse_lyrics(df):
    sw = stopwords.words('english')

    df['cleaned_lyrics'] = df['lyrics'].apply(remove_between_brackets)
    df['cleaned_lyrics'] = df['cleaned_lyrics'].str.lower()
    df['cleaned_lyrics'] = df['cleaned_lyrics'].apply(remove_stopwords)
    df['tokenized_text'] = df["cleaned_lyrics"].apply(word_tokenize)

    return df

def read_lyrics(
    filepath:str = 'data/song_lyrics.csv', 
    n_rows:int = -1, verbose:bool = True
):
    """
    Description
    ----------
    This function imports the lyrics data file and performs basic cleaning 
    operations on it

    Inputs
    ----------
    filepath = The relative path to the file
    n_rows = The number of rows to return from the file
    even_genre_sample = If true, takes an approximately even sample across all genres
    verbose = If true, prints useful intermediates

    Returns
    ----------
    df = A pandas dataframe containing the cleaned lyrics
    """
    # read data
    df = pd.read_csv(filepath, nrows = n_rows)

    # (optional) filter out non-english
    df = df.filter('language == "en').reset_index(drop = True)

    # apply basic cleaning
    df = cleanse_lyrics(df)

    if verbose:
        print(f'Lyrics Dataset: Shape = {df.shape}')
        print(f'\tColumns = {df.columns.tolist()}')

    return df

def read_lyrics_dask(
    filepath:str = 'data/song_lyrics.csv',
    exclude_non_english:bool = True,
    resample_genres:bool = True,
    save_data:bool = True,
    verbose:bool = True
):
    """
    Description
    ----------
    This function works through the entire large file (~9GB) of lyrics using
    dask rather than pandas. It then applies some trimming operations before
    ultimately returning it as a pandas dataframe.

    Inputs
    ----------
    filepath = The relative path to the file
    exclude_non_english = If true, fitlers out non-english words
    resample_genres = If true, resamples records up to the median of count
        songs by genre. This way we don't wind up with extreme unbalance.
    save_data = If true, saves the re-sampled data back to the data folder
    verbose = If true, prints useful intermediates

    Returns
    ----------
    df = A pandas dataframe containing our lyrics
    """
    # read the dataset with dask
    df = dd.read_csv(filepath)

    # (optional) filter out non-english
    if exclude_non_english:
        df = df[df['language'] == 'en']

    # trim to only columns we care about
    cols = ['lyrics', 'genre']
    df = df.rename(columns = {'tag': 'genre'})[cols]
    
    # (optional) resample to median of genre groups
    if resample_genres:
        # retrieve genre record counts and define median
        # (convert to pandas for aggregation)
        genre_counts = df.groupby('genre').size().compute()
        median_count = int(genre_counts.median())

        # define how we will be resampling
        def sample_genre(group, median_count):
            sample_size = min(len(group), median_count) # take min of available or median_count
            return group.sample(sample_size, random_state = 42)
        
        # resample up to the median for each tag
        df_sampled = df.groupby('genre').apply(sample_genre, median_count, meta = df)

    # compute the final results
    df_result = df_sampled.compute()

    # (optional) save recomputed results
    if save_data:
        df_result.to_csv('data/song_lyrics_clean.csv', index = False)

    if verbose:
        print(f'Lyrics (Cleaned): Shape = {df_result.shape}')
        if resample_genres:
            print(f'\tGenre Counts: {genre_counts}')

    return df_result