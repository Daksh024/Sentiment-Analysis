import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Read the Data for training
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Label Encode the column for preprocessing
def label_encode(df,col_name):
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    le.fit(df[col_name])

    return le

# Train Test Split
def train_test_split(df,test_size=0.2):
    from sklearn.model_selection import train_test_split

    return train_test_split(df,test_size=test_size,random_state=42)

# Vectoriing the text
def vectorize(train_x,test_x):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(stop_words='english')
    train_x_vector = tfidf.fit_transform(train_x)
    test_x_vector = tfidf.transform(test_x)
    
    return tfidf, train_x_vector, test_x_vector


class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass

def best_estimator_finder(train_X, train_y):
    # Create a pipeline
    pipe = Pipeline([('clf', DummyEstimator())]) # Placeholder Estimator

    # Candidate learning algorithms and their hyperparameters
    search_space = [
        # {'clf': [DecisionTreeClassifier()],  # Actual Estimator
        #          'clf__criterion': ['gini', 'entropy'],
        #          'clf__max_depth' : [None,5,10],
        #          'clf__max_leaf_nodes' : [None,2,4,8],},
                
                {'clf': [SVC()],
                    'clf__C': [4],
                    'clf__break_ties': [False],
                    'clf__cache_size': [200],
                    'clf__class_weight': [None],
                    'clf__coef0': [0.0],
                    'clf__decision_function_shape': ['ovr'],
                    'clf__degree': [3],
                    'clf__gamma': ['scale'],
                    'clf__kernel': ['rbf'],
                    'clf__max_iter': [-1],
                    'clf__probability': [False],
                    'clf__random_state': [None],
                    'clf__shrinking': [True],
                    'clf__tol': [0.001],
                    'clf__verbose': [False]}
               ]
    
    # Create grid search 
    gs = GridSearchCV(pipe, search_space, verbose=3)
    
    gs.fit(train_X, train_y)
    
    return gs.best_estimator_, gs.best_params_

def save_model(model):
    from joblib import dump
    dump(model, 'best_estimator.joblib')

def load_model(model_path='best_estimator.joblib'):
    from joblib import load
    return load(model_path)

def model_data():
    df = read_data('./IMDB Dataset.csv')
    le= label_encode(df,'sentiment')
    train,test = train_test_split(df,test_size=0.2)
    vectorizer, train_x, test_x = vectorize(train['review'],test['review'])

    import pickle
    pickle.dump(le,open('le.pickle','wb'))
    return le, vectorizer

if __name__ == "__main__":

    df = read_data('./IMDB Dataset.csv')
    le= label_encode(df,'sentiment')
    train,test = train_test_split(df,test_size=0.2)
    vectorizer, train_x, test_x = vectorize(train['review'],test['review'])

    model, params = best_estimator_finder(train_x, le.transform(train['sentiment']))

    print(model)
    print(params)

    save_model(model)