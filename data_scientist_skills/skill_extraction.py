from data_scientist_skills.skills_and_stopwords import all_words_to_remove
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

def remove_more_stopwords(text):
    """Removes unnecessary words from text. Use .apply(remove_more_stopwords) on required df column"""
    text.split(" ")
    clean_text = re.sub(r'\b(%s)\b' % '|'.join(all_words_to_remove), '', text)
    clean_text.replace(',', '')
    return clean_text

def tdidf(X, min_df=0.1, max_df=0.6, results_df=False, transform=False):
    """Performs TD-IDF on pd Series. Results_df=True to return table of results. """
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df)

    if transform:
        # tfidf.fit(X)
        return tfidf

    X_vec = tfidf.fit_transform(X)
    res = pd.DataFrame(X_vec.toarray(), columns = tfidf.get_feature_names_out())
    if results_df:
        return res
    words = (res.sum().sort_values(ascending=False))
    return words
