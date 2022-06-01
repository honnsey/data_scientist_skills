import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from data_scientist_skills.skill_extraction import tdidf

def known_level_encoder(df):
    '''
    Encoding levels based on job titles or required years of experience.
    Job title takes precedence.
    Return a dataframe of 3 level-encoded features.
    '''

    # define combination of strings to search for each level
    junior = 'jr|jnr|junior|grad|entry|trainee|intern'
    senior = 'senior|snr|lead'
    mid = 'mid'

    levels = {'junior': junior, 'senior': senior, 'mid-level': mid}

    temp = df.copy()
    # Search and encode for level in job title
    for k, v in levels.items():
        temp['title',k] = temp['cleaned_title'].str.contains(v)
        temp['title',k] = temp['title',k].apply(lambda x: 1 if x == True else 0)

    # Search through years of experience
    title_columns = [('title', level) for level in levels.keys()]

    condition = (temp[title_columns[0]] == 0) & \
                (temp[title_columns[1]] == 0) & \
                (temp[title_columns[2]] == 0) & \
                (temp['experience'].notnull())

    temp[title_columns[0]][condition] = temp['experience'][condition].apply(lambda x: 1 if x <=  2 else 0)
    temp[title_columns[1]][condition] = temp['experience'][condition].apply(lambda x: 1 if x > 4 else 0)
    temp[title_columns[2]][condition] = temp['experience'][condition].apply(lambda x: 1 if x in range(3,5) else 0)

    return temp

def job_type_encoder(df):
    '''Returns encoding for three streams:
    analyst, engineer and scientist based on job title
    '''

    titles = ['analyst', 'engineer', 'scien']
    tempt_df = pd.DataFrame()

    # Include "encoded" in column name, as keywords extracted from job
    # description may include words in titles
    for title in titles:
        tempt_df['encoded',title] = df['cleaned_title'].str.contains(title)\
                    .apply(lambda x: 1 if x is True else 0)
    return tempt_df

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

    imputer = KNNImputer(n_neighbors= n_neighbors)
    X_imputed = imputer.fit_transform(df)

    return pd.DataFrame(X_imputed, columns= df.columns)

def level_encoder_post_impute(df, threshold= 0.7):
    '''Categorize seniority level based on imputed value (probability)
    return dataframe with 0 or 1 encoded levels'''

    temp_df = df.copy()
    columns = [('title', 'junior'), ('title', 'senior')]
    for column in columns:
        temp_df[column] = df[column].apply(lambda x: 1 if x >= threshold else 0)

    mid_level_filter = (df[columns[0]] < threshold) & (df[columns[1]] < threshold)
    temp_df['title', 'mid-level'] = np.where(mid_level_filter, 1, 0)

    return temp_df

def get_position(df):
    '''
    Return the Category (level and stream) as a Series
    Use to visualise dataframe - to compare with unsupervised model results
    '''

    levels = [('title', 'junior'), ('title', 'senior'), ('title', 'mid-level')]
    streams = [('encoded', 'analyst'), ('encoded', 'engineer'), ('encoded', 'scien')]

    def get_level(row):
        '''Return job level: junior, mid-level or senior'''
        return levels[np.argmax(row.values)][1]

    def get_stream(row):
        '''Return stream: scientist, analyst, engineer'''
        return streams[np.argmax(row.values)][1]

    Category = df[streams].apply(get_stream, axis= 1) \
                        + ' ' \
                        + df[levels].apply(get_level, axis= 1)

    return Category

def process_dataframe(df):
    '''
    Combine all data processing methods for ease of use.
    Input dataframe MUST have cleaned job titles and double-cleaned descriptions.
    Output df = encoded levels & streams + vectorised keywords + category
    '''
    # Step 1: encode levels based on job titles
    known_level_encoded_df = known_level_encoder(df)

    # Step 2: replace 0 with np.nan - prep for imputing
    pre_impute_df = replace_with_NaN(known_level_encoded_df)

    # Step 3: Vectorized job description - return df with vectorised keywords
    vectorised_desc = tdidf(df.cleaned_description, results_df= True)

    # Step 4: Impute NaN based on vectorised description
    temp_df = pd.concat([pre_impute_df,vectorised_desc], axis = 1)
    imputed_vectorised_df = imputer(temp_df) # output df columns = levels and vectorised kws

    # Step 5: Convert probability of levels to 1(True) or 0(false)
    encoded_df = level_encoder_post_impute(imputed_vectorised_df)

    # Step 6: merge levels and streams dataframe
    level_stream_df = pd.concat([job_type_encoder(df),
                                 encoded_df], axis= 1)

    # Step 7: Return new dataframe
    return pd.concat([level_stream_df,
                      get_position(level_stream_df).to_frame(name= "Category")],
                     axis = 1)
