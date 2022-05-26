import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def known_level_encoder(df):
    '''Encoding levels based on job titles.
    Return original df plus 3 level-encoded features.
    '''

    # define combination of strings to search for each level
    junior = 'jr|jnr|junior|grad|entry|trainee|intern'
    senior = 'senior|snr|lead'
    mid = 'mid'

    levels = {'junior': junior, 'senior': senior, 'mid-level': mid}

    # Search and encode for level in job title
    for k, v in levels.items():
        df['title',k] = df['cleaned_title'].str.contains(v)
        df['title',k] = df['title',k].apply(lambda x: 1 if x == True else 0)

    return df

def job_type_encoder(df):
    '''Returns encoding for three streams:
    analyst, engineer and scientist based on job title
    '''

    titles = ['analyst', 'engineer', 'scien']

    # Include "encoded" in column name, as keywords extracted from job
    # description may include words in titles
    for title in titles:
        df[f'encoded {title}'] = df['cleaned_title'].str.contains(title)\
                    .apply(lambda x: 1 if x is True else 0)
    return df

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
    salary['mean_salary'] = (salary['low'] + salary['high']) / 2

    #dropping unwanted columns
    to_drop = ['flag']
    if only_mean:
        to_drop.extend(['low','high'])

    salary.drop(columns = to_drop, inplace = True)
    temp_df = temp_df.join(salary)
    temp_df.drop(columns = 'salary_estimate', inplace = True)

    return temp_df

def replace_with_NaN(df):
    '''Replace unknown job title with NaN.
    Data prepartion for KNNImputer.
    Return the entire dataframe with updated values'''

    columns = [('title', 'junior'), ('title', 'senior'), ('title', 'mid-level')]

    filters = (df['title', 'junior'] == 1)& (df['title', 'senior'] == 0)& (df['title', 'mid-level'] == 0)| \
            (df['title', 'junior'] == 0)& (df['title', 'senior'] == 1)& (df['title', 'mid-level'] == 0)| \
            (df['title', 'junior'] == 0)& (df['title', 'senior'] == 0)& (df['title', 'mid-level'] == 1)

    df[columns] = df[columns].where(filters)

    return df

def imputer(df, n_neighbors= 5):
    '''Impute Nan Values in dataframe
    Return dataframe with updated values'''

    # Drop first 13 columns
    # dropped columns = ['job_title', 'job_description', 'rating', 'company_name', 'location',
    #    'headquarters', 'size', 'founded', 'type_of_ownership', 'industry',
    #    'sector', 'cleaned_description', 'cleaned_title'],

    X = df.drop(columns= df.columns[0:13])

    imputer = KNNImputer(n_neighbors= n_neighbors)
    X_imputed = imputer.fit_transform(X)

    return pd.DataFrame(X_imputed, columns= X.columns)

def level_encoder_post_impute(df, threshold= 0.7):
    '''Categorize seniority level based on imputed value (probability)
    return dataframe with 0 or 1 encoded levels'''

    columns = [('title', 'junior'), ('title', 'senior')]
    for column in columns:
        df[column] = df[column].apply(lambda x: 1 if x >= threshold else 0)

    mid_level_filter = (df[columns[0]] < threshold) & (df[columns[1]] < threshold)
    df['title', 'mid-level'] = np.where(mid_level_filter, 1, 0)

    return df

def get_position(df):
    '''
    Return the Category (level and stream) as a Series
    Use to visualise dataframe - to compare with unsupervised model results
    '''

    levels = [('title', 'junior'), ('title', 'senior'), ('title', 'mid-level')]
    streams = ['encoded analyst', 'encoded engineer', 'encoded scien']

    def get_level(row):
        '''Return job level: junior, mid-level or senior'''
        return levels[np.argmax(row.values)][1]

    def get_stream(row):
        '''Return stream: scientist, analyst, engineer'''
        return streams[np.argmax(row.values)]

    Category = df[streams].apply(get_stream, axis= 1) \
                        + ' ' \
                        + df[levels].apply(get_level, axis= 1)

    return Category
