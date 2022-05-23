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
    df = get_description(dataframe)
    df['cleaned_description'] = df['job_description'].apply(clean)
    return pd.DataFrame(df['cleaned_description'])

def process_salary_estimate(dataframe, only_mean = True):
    """Inteprets the salary_estimate string; Returns the mean of the range.
    Returned values are in units of 1k. If the string contains ' Per Hour', the
    salary will be calculated as 40hours/week, 52weeks/year. If only_mean is set
    to False, the returned DataFrame will include the low and high of the range
    as separate columns.
    """
    salary = dataframe['salary_estimate']
    temp_df = dataframe.copy()

    #removes extra characters  ### This section needs to be written handle salary ranges dynamically -JP
    salary = salary.apply(lambda x: x.replace('K', ''))\
            .apply(lambda x: x.replace('$', ''))\
            .apply(lambda x: x.replace('(Glassdoor est.)', ''))\
            .apply(lambda x: x.replace('(Employer est.)', ''))

    #splits into a low and high column
    salary = salary.str.split("-", expand = True)
    salary.rename(columns = {0:'low', 1:'high'}, inplace = True)

    #fills in the empty values, some lows are blank
    salary = salary.apply(lambda x: x.replace('', '0'))

    #removes the ' Per Hour' string and adds a flag  ### Dynamic version would check for any mention of 'hour'/'hr' -JP
    salary['high'] = salary['high'].apply(lambda x: x.replace(' Per Hour', '-*'))
    salary[['high','flag']] = salary['high'].str.split("-", expand = True)

    #casts values to numerical
    salary[['low','high']] = salary[['low','high']].astype(int)

    #converts the hourly wages to a yearly salary  ### There is a better calculations than my assumptions here -JP
    salary.loc[salary.flag == '*', ['low','high']] =\
        salary.loc[salary.flag == '*', ['low','high']]*40*52/1000

    #creates the mean column
    salary['mean'] = (salary['low'] + salary['high']) / 2

    #dropping unwanted columns
    to_drop = ['flag']
    if only_mean:
        to_drop.extend(['low','high'])

    salary.drop(columns = to_drop, inplace = True)
    temp_df = temp_df.join(salary)
    temp_df.drop(columns = 'salary_estimate', inplace = True)

    return temp_df
