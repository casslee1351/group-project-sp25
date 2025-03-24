import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd

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
    filepath:str = 'song_lyrics.csv', 

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
    verbose = If true, prints useful intermediates

    Returns
    ----------
    df = A pandas dataframe containing the cleaned lyrics
    """
    # read data
    df = pd.read_csv(filepath, nrows = n_rows)

    # apply basic cleaning
    df = cleanse_lyrics(df)

    if verbose:
        print(f'Lyrics Dataset: Shape = {df.shape}')
        print(f'\tColumns = {df.columns.tolist()}')

    return df