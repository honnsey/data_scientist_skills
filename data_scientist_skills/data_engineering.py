import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from data_scientist_skills.skill_extraction import tdidf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

    return temp.iloc[:,-3:]

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

###Sector Encoding Section
# As currently written, these methods do no drop the original encoded column

def make_sector_group_column(df, give_group_profiles = False):
    """Returns a df with an appended column that has the sector group number.
    --Unimplemented--
    A list of dictionaries with profiles of each group can be gotten by setting
    give_group_profiles to True.
    """
    ## Groups were first made using a standard method of grouping sectors found
    ## here: https://www.investopedia.com/terms/s/sector.asp
    ## However this led to unbalanced group sizes so I rearranged them in a way
    ## hopefully sensible -JP
    ## Currently they are distributed as follows:
    ## [308, 713, 671, 674, 1694, 1203]

    #Can benefit from OOP, groups can be a class object for easy modification.
    # Further, it would help to untangle the features made ready for model to
    # make results interpretable for the end user

    temp_df = df.copy()
    groups = [
    ['Mining & Metals', 'Agriculture & Forestry', 'Transportation & Logistics',
     'Construction, Repair & Maintenance', 'Oil, Gas, Energy & Utilities',
     'Manufacturing', 'Aerospace & Defense',],

    ['Accounting & Legal', 'Insurance','Finance', ],

    [ 'Telecommunications', 'Media', 'Real Estate', 'Travel & Tourism',
     'Arts, Entertainment & Recreation','Restaurants, Bars & Food Services',
     'Retail', 'Non-Profit','Government','Education','Consumer Services',],

    ['Health Care','Biotech & Pharmaceuticals',],

    ['Information Technology',],

    ['Business Services',]

    ]

    #Creates a list with a True value where the sector is found in a group
    #Its index+1 is then assigned as the value in the new sector_group column
    #Those not found (i.e. the -1) are assigned -1 in the new column for easy
    #filtering during encoding
    func = lambda sector: [sector in group for group in groups].index(True) + 1\
                            if any(sector in group for group in groups)\
                            else -1

    #Applies previous function in mapping to the new sector_group columns
    temp_df['sector_group']= temp_df.sector.apply(func)

    return temp_df

def label_encoder(df, column):
    """Alternative method that utilizes LabelEncoder that does no grouping.
    Function exits if passed column is not found in dataframe
    """
    ### Can benefit from OOP, LabelEncoder has inverse transform, so values can
    ### be reverted to the human readable text if the fit transfoermer is kept -JP

    temp_df = df.copy()

    #Checks that passed column is a column in the df
    try:
        temp_df[column]
    except:
        raise(Exception(f"{column} isn't found in passed data frame"))

    #Creates and fits label encoder
    label_enc = LabelEncoder()
    label_enc.fit(temp_df[column])

    #Label encodes passed column into a new column
    temp_df[f'lenc_{column}'] = label_enc.transform(temp_df[column])

    return temp_df

def impute_empty_sectors(df_with_sector_group = ''):
    """Intended to impute job listings with empty sector values. Can be done
    simply by probablity distribution. Next step in complexity would be a regex
    or similar word matching (RapidFuzz) from words found in company name,
    description, and job title potentially. Too many choices for now so left
    empty -JP
    """
    pass

def one_hot_encode_sector_groups(df, col = 'sector_group', drop = 'first'):
    """OHE a given column, by default the sector_group column.
    By default the -1 values are not encoded.
    Set drop to None if the -1 values are to be encoded.
    OHE drop works consistently for numericals, beware/set drop to None if
    values are non-numerical or the column lacks null -1
    """
    ###Can benefit from OOP, fit model could be kept to inverse transform encoded
    ###values. Written that the method can be generalized for encoding other columns
    ###Further this could be generalized again to handle multiple column encodings
    ### -JP

    temp_df = df.copy()

    #Checks that passed column is a column in the df
    try:
        temp_df[col]
    except:
        raise(Exception(f"'{col}' isn't found in passed data frame"))

    #Creates and fits the encoder
    ohe = OneHotEncoder(sparse=False, drop = drop)
    ohe.fit(temp_df[[col]])

    #Creates the names for the encoded columns. The ternary expression keeps
    #the first value (i.e. -1) if drop is set to None
    columns = list(ohe.categories_[0][1:]) if drop else list(ohe.categories_[0][:])
    columns = [f'{col}_{column}' for column in columns]

    #Creates encoded columns and values dataframe
    encoded_df = pd.DataFrame(ohe.transform(temp_df[[col]]), columns = columns)

    #Joins encoded columns to original dataframe and returns new dataframe
    return temp_df.join(encoded_df)
