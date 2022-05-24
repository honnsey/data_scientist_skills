def confirmed_level_df(df):
    '''Return dataframe with levels stated in job title.
    Classification encoded for "junior", "mid-level" and "senior".
    Encoder applicable for cleaned job title
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

    # output dataframe with confirmed job titles only
    output_df = df[(df['title','junior'] == 1) |
                   (df['title','mid-level'] == 1) |
                   (df['title','senior'] == 1)]

    return output_df
