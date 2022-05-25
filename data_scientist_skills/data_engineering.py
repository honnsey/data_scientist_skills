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
