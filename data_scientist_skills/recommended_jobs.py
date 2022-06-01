import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
from data_scientist_skills.data import clean_dataframe, get_data
from data_scientist_skills.data_engineering import process_dataframe

def prep_input():
    '''
    Preparing inputs for recommended_jobs model.
    Input = dataframe with 15 top skills extracted from job description
    '''
    df = get_data()
    clean_df = clean_dataframe(df)
    processed_df = process_dataframe(clean_df)
    skills_df = processed_df.drop(columns = [('encoded', 'analyst'), ('encoded', 'engineer'),
          ('encoded', 'scien'),     ('title', 'junior'),
           ('title', 'senior'),  ('title', 'mid-level'), 'Category'])

    # binarize skills
    bin_skills_df = skills_df.copy()
    bin_skills_df[bin_skills_df > 0] = 1

    # select 15 skills that appear the most
    appearance_threshold = bin_skills_df.sum().sort_values()[-15]

    # filter dataframe down to only include skill of interest
    cols_to_keep = list(np.where(bin_skills_df.sum().values > appearance_threshold)[0])
    return bin_skills_df.iloc[:, cols_to_keep], cleaned_df

class recommend_jobs():
    ## X = skills of interest
    def __init__(self):
        self.X, self.df = prep_input()
        self.model = NearestNeighbors().fit(self.X)


    def predict(self, input):
        user_skills_df = pd.DataFrame(input)
        recommended_job_ids = list(self.model.kneighbors(user_skills_df,
                                                            5,
                                                            return_distance=False)[0])
        return self.df.iloc[recommended_job_ids, :]
