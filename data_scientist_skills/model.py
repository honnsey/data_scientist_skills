from data_scientist_skills.data import get_data, clean_dataframe
from data_scientist_skills.data_engineering import process_dataframe, label_encoder
from data_scientist_skills.skill_extraction import tdidf, remove_more_stopwords
from sklearn.svm import SVC
import pandas as pd

def data_prep(return_x=False):
    """Used when instantiating Model class"""
    # Data preparation for model training
    df = get_data()
    clean_df = clean_dataframe(df)
    processed_df = process_dataframe(clean_df)

    # X and y
    X = clean_df['cleaned_description'] + df['cleaned_title']

    # Returns X for fitting vectorizer, to transform input
    if return_x:
        return X

    X_svc = tdidf(X, results_df=True)
    y_svc = label_encoder(processed_df, 'Category')['lenc_Category']
    return X_svc, y_svc

def model_prep():
    """Used in Model class constructor to fit model on training data"""
    # Get X & y
    X_svc, y_svc = data_prep()
    # Model
    model_svc = SVC(kernel='poly', C=0.8, coef0=0.7)
    model_svc.fit(X_svc, y_svc)

    return model_svc

def init_vectorizer():
    """Used in Model class constructor to initialize vectorizer for input"""
    X = data_prep(return_x=True)
    vectorizer = tdidf(X, transform=True)
    vectorizer.fit(X)
    return vectorizer

class Model:
    """Import Model, instantiate model = Model(), then model.predict(Job description string or job skills list)"""
    def __init__(self):
        self.model = model_prep()
        self.vectorizer = init_vectorizer()

    def prep_input(self, input):
        # Vectorize input >>> Pick's out key words from input and vectorizes
        vectorized_input = self.vectorizer.transform(input)

        # Turns vectorized input into DataFrame for input into prediction model
        input_dataframe = pd.DataFrame(vectorized_input.toarray())

        return input_dataframe

    def predict(self, input):
        if type(input) == str:
            input = [input]
        input_ready = self.prep_input(input)

        prediction = self.model.predict(input_ready)

        titles = ['Data Analyst - Junior',
                  'Data Analyst - Intermediate',
                  'Data Analyst - Senior',
                  'Data Engineer - Junior',
                  'Data Engineer - Intermediate',
                  'Data Engineer - Senior',
                  'Data Scientist - Junior',
                  'Data Scientist - Intermediate',
                  'Data Scientist - Senior']
        return titles[int(prediction)]
