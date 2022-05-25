import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def level_encoder(df):
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
    analyst, engineer and scientist
    '''

    titles = ['analyst', 'engineer', 'scien']

    for title in titles:
        df[title] = df['cleaned_title'].str.contains(title)\
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


###Sector Encoding Section

def make_sector_group_column(df, give_group_profiles = False):
    """Returns a df with an appended column that has the sector group number.
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
    ['Mining & Metals', 'Agriculture & Forestry', 'Construction, Repair & Maintenance',
         'Oil, Gas, Energy & Utilities', 'Manufacturing','Aerospace & Defense','Transportation & Logistics',],

    ['Accounting & Legal', 'Insurance','Finance', ],

    [ 'Telecommunications', 'Media','Real Estate','Travel & Tourism',
     'Arts, Entertainment & Recreation','Restaurants, Bars & Food Services','Retail',
     'Non-Profit','Government','Education','Consumer Services',],

    ['Health Care','Biotech & Pharmaceuticals',],

    ['Information Technology',],

    ['Business Services',]

    ]

    #Creates a list with a True value where the sector is found in a group
    #Its index+1 is then assigned as the value in the new sectuor group column
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
    try:
        temp_df[column]
    except:
        raise(Exception(f"{column} isn't found in passed data frame"))

    label_enc = LabelEncoder()
    label_enc.fit(temp_df[column])
    temp_df[f'lenc_for_{column}'] = label_enc.transform(temp_df[column])
    return temp_df

def impute_empty_sectors(df_with_sector_group):
    pass

def one_hot_encode_sector_groups(df, col = 'sector_group', drop = 'first'):
    """OHE a given column, by default the sector_group column.
    By default the -1 values are not encoded.
    Set drop to None if the -1 values are to be encoded.
    """
    ###Can benefit from OOP, fit model could be kept to inverse transform encoded
    ###values. Written that the method can be generalized for encoding other columns
    ### -JP

    temp_df = df.copy()

    #Checks that passed column is a column in the df
    try:
        temp_df[col]
    except:
        raise(Exception(f"'{col}' isn't found in passed data frame"))

    #Creates the encoder
    ohe = OneHotEncoder(sparse=False, drop = drop)
    ohe.fit(temp_df[[col]])

    #Creates the names for the encoded columns. The ternary expression keeps -1
    #if drop is set to None
    columns = list(ohe.categories_[0][1:]) if drop else list(ohe.categories_[0][:])
    columns = [f'{col}_{column}' for column in columns]

    #Creates encoded columns and values dataframe
    encoded_df = pd.DataFrame(ohe.transform(temp_df[[col]]), columns = columns)

    #Joins encoded columns to original dataframe and returns new dataframe
    return temp_df.join(encoded_df)
