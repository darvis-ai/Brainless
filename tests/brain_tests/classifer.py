# """
# nosetests -sv --nologcapture tests/quick_test.py
# nosetests --verbosity=2 --detailed-errors --nologcapture --processes=4 --process-restartworker --process-timeout=1000 tests/quick_test.py
# """

import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# old_version : from brainless import Predictor
from brainless.algorithm.classifier import Classifier

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.abspath(os.path.dirname(os.path.dirname(__file__)))] + sys.path

os.environ['is_test_suite'] = 'True'

# os.environ['KERAS_BACKEND'] = 'theano'




def get_titanic_binary_classification_dataset(basic=True):
    try:
        df_titanic = pd.read_csv(os.path.join('data', 'titanic.csv'))
    except FileNotFoundError:
        print('titanic.csv could not be found, attempting to retrieve from url.')
        dataset_url = 'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/ff414a1bcfcba32481e4d4e8db578e55872a2ca1/titanic.csv'
        df_titanic = pd.read_csv(dataset_url)
        # Do not write the index that pandas automatically creates
        if not os.path.exists('data'):
            os.mkdir('data')
        df_titanic.to_csv(os.path.join('data', 'titanic.csv'), index=False)

    df_titanic = df_titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    if basic:
        df_titanic = df_titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    df_titanic_train, df_titanic_test = train_test_split(
        df_titanic, test_size=0.33, random_state=42)
    return df_titanic_train, df_titanic_test


def classification_test():
    np.random.seed(0)
    # model_name = 'GradientBoostingClassifier'
    model_name = 'LGBMClassifier'

    df_titanic_train, df_titanic_test = get_titanic_binary_classification_dataset()
    df_titanic_train['DELETE_THIS_FIELD'] = 1

    column_descriptions = {
        'survived': 'output',
        'embarked': 'categorical',
        'pclass': 'categorical',
        'sex': 'categorical'
    }

    ml_predictor = Classifier(
        type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    lower_bound = -0.16
    if model_name == 'DeepLearningClassifier':
        lower_bound = -0.245
    if model_name == 'LGBMClassifier':
        lower_bound = -0.225

    assert lower_bound < test_score < -0.135


if __name__ == '__main__':
    classification_test()
