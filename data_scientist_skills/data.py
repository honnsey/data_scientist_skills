import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import unidecode  #This import us unused do we need to have this? -JP
from data_scientist_skills.skill_extraction import remove_more_stopwords

from pathlib import Path
from os import getcwd


def get_data():
    #Get data
    ###removing ../ allows terminal command 'make run_locally' to work -JP
    df1 = pd.read_csv("raw_data/DataAnalyst.csv")
    df2 = pd.read_csv("raw_data/DataScientist.csv")
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

def years_experience(string):
    '''
    Extract number of years of experience required in a job description.
    If multiple requirements present in a description, return maximum value.
    If none found, return NaN.
    '''
    pattern = r"\d{1,2}(?=.{0,5}? years?.{0,3}? experience)"
    experience = re.findall(pattern, string) # return a list of matching numbers found based on expression

    # remove number of years greater than 15 - non-sensical values
    experience = [_ for _ in experience if int(_) <= 15]
    if experience == []:
        return np.nan
    return int(max(experience))

def get_cleaned_description(dataframe):
    """Will return cleaned_description column only"""
    df = get_description(dataframe)
    df['cleaned_description'] = df['job_description'].apply(clean)
    return pd.DataFrame(df['cleaned_description'])

def clean_dataframe(df):
    """Returns dataframe with clean_description and clean_title columns"""
    # Create new column for years of experience extracted from raw job description
    df['experience'] = df['job_description'].apply(years_experience)
    df['cleaned_description'] = df['job_description'].apply(clean)
    df.cleaned_description = df.cleaned_description.apply(remove_more_stopwords)
    df['cleaned_title'] = df['job_title'].apply(clean)
    return df
