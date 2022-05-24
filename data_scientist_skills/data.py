import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import unidecode

def get_data():
    #Get data
    df1 = pd.read_csv("data_scientist_skills/raw_data/DataAnalyst.csv")
    df2 = pd.read_csv("data_scientist_skills/raw_data/DataScientist.csv")
    #Concat to full dataframe
    df = pd.concat([df1, df2], ignore_index=True, axis=0)
    #Drop columns and reformat names
    df.drop(columns=['Unnamed: 0', 'index', 'Revenue', 'Competitors', 'Easy Apply'], inplace=True)
    df.columns = [column.replace(' ', '_').lower() for column in df.columns]
    df.reset_index(drop=True)
    df.drop_duplicates()
    return df

def get_description(df):
    return pd.DataFrame(df['job_description'])

def clean(description):
    ''' Function returns cleaned text from one input string, applicable for description columns and/or title column'''
    stop_words = set(stopwords.words('english'))
    # Remove \n
    description = description.replace('\n', ' ')
    # Remove Whitespace
    description = description.strip()
    # Make lower case
    description = description.lower()
    # Remove accents
    description = unidecode.unidecode(description)
    # Remove Punctuation
    for punctuation in string.punctuation:
        description = description.replace(punctuation, '')
    # Tokenizing
    description_tokenized = word_tokenize(description)
    description_tokenized = [w for w in description_tokenized if not w in stop_words]  # Removing stop words
    # Lemmatize
    lemmatized_description = [WordNetLemmatizer().lemmatize(word, pos="v") # V for Verb
                            for word in description_tokenized]
    cleaned_sentence = ' '.join(word for word in lemmatized_description)
    return cleaned_sentence


def get_cleaned_description(dataframe):
    """Will return cleaned_description column only"""
    df = get_description(dataframe)
    df['cleaned_description'] = df['job_description'].apply(clean)
    return pd.DataFrame(df['cleaned_description'])

def clean_dataframe(df):
    """Returns dataframe with clean_description and clean_title columns"""
    df['cleaned_description'] = df['job_description'].apply(clean)
    df['cleaned_title'] = df['job_title'].apply(clean)
    return df
