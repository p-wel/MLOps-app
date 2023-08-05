"""
This is a boilerplate pipeline
generated using Kedro 0.18.10
"""

import logging
from typing import Dict, Tuple

import pandas as pd
import wandb

from .prepare_data import prepare_raw_data as _prepare_raw_data
import keras
from keras.layers import Dense
from keras.callbacks import CSVLogger

from autogluon.tabular import TabularDataset, TabularPredictor

def prepare_raw_data() -> str:
    logger = logging.getLogger(__name__)
    logger.info("Beginning old&new data merging and preparation")
    datafile = _prepare_raw_data()
    logger.info("Data successfully prepared")
    return datafile


def load_and_split_data(
    filename: str, parameters: Dict[str, any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        filename: name of the file with clean training data
        parameters: config provided from yaml file
    Returns:
        Split data.
    """
    print("Received filename:", filename, "params", parameters)
    training_data = pd.read_csv(filename)
    data_train = training_data.sample(
        frac=parameters["train_fraction"], random_state=parameters["random_state"]
    )
    data_test = training_data.drop(data_train.index)

    X_train = data_train.drop(columns=parameters["target_column"])
    X_test = data_test.drop(columns=parameters["target_column"])
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]
    return X_train, X_test, y_train, y_test


def create_model() -> keras.models.Sequential:
    model = keras.models.Sequential()
    model.add(Dense(255, input_shape=(11,), activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model: keras.models.Sequential, x_train: pd.DataFrame, y_train: pd.Series) -> (keras.models.Sequential, str):
    model.fit(x_train, y_train, epochs=25, callbacks=CSVLogger(r'data/08_reporting/traning_report.csv'), verbose=2)

    return model, 'data/08_reporting/traning_report.csv'


def train_model_automl(x_train: pd.DataFrame, y_train: pd.Series) -> TabularPredictor:
    training_data = x_train.copy()
    feature_column = 'stroke'
    training_data[feature_column] = y_train

    predictor = TabularPredictor(label=feature_column, verbosity=2)
    predictor.fit(training_data, time_limit=30)
    lead = predictor.leaderboard()
    wandb.login(key="022cc496d7da513863890bbcc7abe80098ee91d9")
    wandb.init(project="autogluon")
    for row in lead.iterrows():
        wandb.log(dict(row[1]))
    wandb.finish()
    return predictor


def send_data_to_wandb(filename):
    wandb.login(key="022cc496d7da513863890bbcc7abe80098ee91d9")
    wandb.init(project="iron ore")
    data = pd.read_csv(filename)
    for row in data.iterrows():
        wandb.log(dict(row[1]))
    wandb.finish()


def make_predictions(model,
    X_test: pd.DataFrame, y_test: pd.Series
) -> pd.Series:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """
    wandb.login(key="022cc496d7da513863890bbcc7abe80098ee91d9")
    run = wandb.init(project="Model Registry")
    
    loss, accuracy = model.evaluate(X_test, y_test)
    wandb.log({"train/loss":loss,
              "train/accuracy": accuracy})
    art = wandb.Artifact(f"HeartStroke_model_Sequential-{wandb.run.id}", 
                        type="model",
                        metadata={'loss': loss,
                                  'accuracy': accuracy,
                                  'model':'Sequential'})

    model.save("model")
    art.add_file("model/saved_model.pb")
    wandb.log_artifact(art, aliases=["latest"])
    assert isinstance(model, TabularPredictor) or isinstance(model, keras.models.Sequential), "Invalid model"
    print("Test data", X_test)
    
    answers = model.evaluate(X_test,y_test)
    print(answers)


    return answers


def report_accuracy(results):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    accuracy = results[1]
    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
