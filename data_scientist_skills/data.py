import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import unidecode

def get_data(df1_dir, df2_dir): #Relative path to each data file
    #Get data
    df1 = pd.read_csv(df1_dir)
    df2 = pd.read_csv(df2_dir)
    #Drop Columns
    df1.drop(columns='Unnamed: 0', inplace=True)
    df2.drop(columns = ['Unnamed: 0', 'index'], inplace = True)
    #Concat to full dataframe
    df = pd.concat([df1, df2], ignore_index=True, axis=0)

    return df

def get_description(df):
    return pd.DataFrame(df['Job Description'])

def clean(description):   # .apply to description or title column
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
    df = get_description(dataframe)
    df['Cleaned Description'] = df['Job Description'].apply(clean)
    return pd.DataFrame(df['Cleaned Description'])
