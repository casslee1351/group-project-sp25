import re
# from nltk.corpus import stopwords 
# from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

import numpy as np
import pandas as pd
from collections import Counter
from statistics import median

import dask.dataframe as dd

# def remove_between_brackets(text):
#   """Removes all text between any matching pair of brackets, including the brackets themselves."""
#   return re.sub(r'\[.*?\]', '', text)

# def cleanse_lyrics(df):
#     sw = stopwords.words('english')

#     df['cleaned_lyrics'] = df['lyrics'].apply(remove_between_brackets)
#     df['cleaned_lyrics'] = df['cleaned_lyrics'].str.lower()
#     df['cleaned_lyrics'] = df['cleaned_lyrics'].apply(remove_stopwords)
#     df['tokenized_text'] = df["cleaned_lyrics"].apply(word_tokenize)

#     return df

# def read_lyrics(
#     filepath:str = 'data/song_lyrics.csv', 
#     n_rows:int = -1, verbose:bool = True
# ):
#     """
#     Description
#     ----------
#     This function imports the lyrics data file and performs basic cleaning 
#     operations on it

#     Inputs
#     ----------
#     filepath = The relative path to the file
#     n_rows = The number of rows to return from the file
#     even_genre_sample = If true, takes an approximately even sample across all genres
#     verbose = If true, prints useful intermediates

#     Returns
#     ----------
#     df = A pandas dataframe containing the cleaned lyrics
#     """
#     # read data
#     df = pd.read_csv(filepath, nrows = n_rows)

#     # (optional) filter out non-english
#     df = df.filter('language == "en').reset_index(drop = True)

#     # apply basic cleaning
#     df = cleanse_lyrics(df)

#     if verbose:
#         print(f'Lyrics Dataset: Shape = {df.shape}')
#         print(f'\tColumns = {df.columns.tolist()}')

#     return df

# === Take Two ===

def clean_lyrics(lyric):
    """
    Removes structural bits of the lyrics
    TODO Remove punctuations.
    """
    # remove structural bits
    lyric = re.sub(r"\[.*?\]", "", lyric)
    lyric = re.sub(r"[^a-zA-Z0-9\s]", "", lyric).lower()

    # remove extra white space
    lyric = " ".join(lyric.split())
    
    return lyric

def read_and_clean_raw_lyrics(
    filepath:str = 'data/song_lyrics.csv',
    n_rows = 'All',
    exclude_non_english:bool = True,
    resample_genres:bool = True,
    save_data:bool = True,
    verbose:bool = True
):
    """
    Description
    ----------
    This function reads the large lyrics file and parses it down to only
    english lyrics, and it resamples the genres so it's more well balanced.

    Inputs
    ----------
    filepath = The relative path to the file
    n_rows = If 'All', then we read in all rows. Otherwise we read in an int
        number of rows from the filepath.
    exclude_non_english = If true, fitlers out non-english words
    resample_genres = If true, resamples records up to the median of count
        songs by genre. This way we don't wind up with extreme unbalance.
    save_data = If true, saves the re-sampled data back to the data folder
    verbose = If true, prints useful intermediates

    Returns
    ----------
    df = A pandas dataframe containing our lyrics
    """
    assert n_rows is 'All' or isinstance(n_rows, int)

    # read the dataset with dask
    if n_rows is not None:
        df = pd.read_csv(filepath, nrows = n_rows)
    else:
        df = pd.read_csv(filepath)

    # (optional) filter out non-english
    if exclude_non_english:
        df = df[df['language'] == 'en']    

    # trim to only columns we care about
    cols = ['lyrics', 'genre']
    df = df.rename(columns = {'tag': 'genre'})[cols]
       
    # (optional) resample to median of genre groups
    if resample_genres:
        # retrieve genre record counts and define median
        genre_counts = df['genre'].value_counts()
        median_count = int(median(genre_counts.to_dict().values()))
        if verbose:
            print(f'Genre Counts Before Resampling:')
            for k, v in genre_counts.to_dict().items():
                print(f'\t{k}: {v}')

        # define how we will be resampling
        def sample_genre(group, median_count):
            sample_size = min(len(group), median_count) # take min of available or median_count
            return group.sample(sample_size, random_state = 42)
        
        # resample up to the median for each tag
        df = df.groupby('genre').apply(sample_genre, median_count).reset_index(drop = True)

        # recount resampled data
        if verbose:
            genre_counts_resampled = df['genre'].value_counts()
            print(f'\nGenre Counts After Resampling:')
            for k, v in genre_counts_resampled.to_dict().items():
                print(f'\t{k}: {v}')
            print()

    # clean up strings of lyrics
    df['lyrics'] = df['lyrics'].apply(lambda x: clean_lyrics(x))

    # (optional) save recomputed results
    if save_data:
        df.to_csv(f'data/song_lyrics_clean.csv', index = False)

    if verbose:
        print(f'Lyrics (Cleaned): Shape = {df.shape}')
        print(f'\tColumns = {df.columns.tolist()}')

    return df

def read_cleaned_lyrics(
    filepath:str = 'data/song_lyrics_clean.csv',
    n_rows_per_genre = 'All',
    verbose:bool = True
):
    """ 
    Description
    ----------
    This function reads in the data that was cleaned by the 
    read_and_clean_raw_lyrics.

    Inputs
    ---------
    filepath = The path tot he file we want to read in
    n_rows_per_genre = The number of rows per genre that we want to 
        retrieve. If 'All', retrieves all. Otherwise it returns the 
        max of n_rows_per_genre and number of records available.
    verbose = If true, prints useful intermediates

    Returns
    ----------
    df = A pandas dataframe containg the lyrics and genre
    genre_mapping = A dict containing the mapping of idx to genre.
        Used downstream for preprocessing and then for final prediction.
    """
    assert n_rows_per_genre == 'All' or isinstance(n_rows_per_genre, int)

    # read data
    df = pd.read_csv(filepath)

    # (optional) trim to n_rows_per_genre
    if isinstance(n_rows_per_genre, int):
        # retrieve genre record counts and define median
        genre_counts = df['genre'].value_counts()
        if verbose:
            print(f'Genre Counts Before Resampling:')
            for k, v in genre_counts.to_dict().items():
                print(f'\t{k}: {v}')

        # define how we will be resampling
        def sample_genre(group, n_rows_per_genre):
            sample_size = min(len(group), n_rows_per_genre) # take min of available or median_count
            return group.sample(sample_size, random_state = 42)
        
        # resample up to the median for each tag
        df = df.groupby('genre').apply(sample_genre, n_rows_per_genre).reset_index(drop = True)

        # recount resampled data
        if verbose:
            genre_counts_resampled = df['genre'].value_counts()
            print(f'\nGenre Counts After Resampling:')
            for k, v in genre_counts_resampled.to_dict().items():
                print(f'\t{k}: {v}')
            print()

    # define genre mapping
    genres = df['genre'].astype('category')
    genre_mapping = {idx: label for idx, label in enumerate(genres.cat.categories)}
    genre_mapping

    if verbose:
        print(f'Cleaned Lyrics: Shape = {df.shape}')
        print(f'\tColumns = {df.columns.tolist()}')
        print(f'Genre Mapping = {genre_mapping}')

    return df, genre_mapping