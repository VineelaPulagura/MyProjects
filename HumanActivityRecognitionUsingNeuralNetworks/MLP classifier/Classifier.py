# Created By    : Vineela Pulagura
# Created on    : 21-01-2021
# Description   : To run the classifier and get the accuracy, plot the data, confusion matrix
import logging
import os
import sys
from datetime import datetime
from logging import getLogger, StreamHandler
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

from Plots import PyPlots
from Utils import UtilityFunctions
from DeclareConstants import Constants


# ________Setting a path to save the log files and figures_________
path = "Logs\\NN_results_"+datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(path)
# ________Configure the format of logs______________________
fmt = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
logging.basicConfig(filename=path+"\\classifier.log", format=fmt, level=logging.DEBUG)
mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
mpl_logger.setLevel(logging.WARNING)
getLogger().addHandler(StreamHandler(sys.stdout))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug(path+"\\classifier.log")


class MlpClassifier:
    logger.debug("In the class Classifier")

# ____________________________Run the classifier_______________________________________________
def run_classifier(parameters: Dict) -> Tuple[float, float]:
    # _______________________Variable declaration____________________________________
    con_mat: Dict[str, np.ndarray] = {"train": [], "valid": [], "test": []}
    precision: Dict[str, List] = {"train": [], "valid": [], "test": []}
    recall: Dict[str, List] = {"train": [], "valid": [], "test": []}
    f1: Dict[str, List] = {"train": [], "valid": [], "test": []}
    accuracy: Dict[str, List] = {"train": [], "valid": [], "test": []}
    loss: Dict[str, List] = {"train": [], "valid": [], "test": []}
    mse: Dict[str, List] = {"train": [], "valid": [], "test": []}

    X_train, X_test, Y_train, Y_test = parameters["X_train"], parameters["X_test"], parameters["Y_train"], parameters["Y_test"]
    labels = UtilityFunctions.swap_last_to_first(np.array(pd.read_table(Constants.LabelData, sep=" ", header=None)))

    logger.debug(f"{X_train.shape=} {X_test.shape=}")
    logger.debug(f"{Y_train.shape=} {Y_test.shape=}")

    # ________replacing the maximum class value to 0 as categorical data starts with 0________________
    Y_test[Y_test == 12] = 0
    Y_train[Y_train == 12] = 0

    # ________Convertiing to categorical data(vector representation in binary matrix format)________________
    Y_test = tf.keras.utils.to_categorical(Y_test)

    # __________________Splitting the data into n_splits____________________________
    sk = StratifiedKFold(n_splits=parameters["no_of_folds"], shuffle=True, random_state=71)
    for index, (train_index, test_index) in enumerate(sk.split(X_train, Y_train)):
        X_tr, X_val = X_train[train_index], X_train[test_index]
        Y_tr, Y_val = Y_train[train_index], Y_train[test_index]
        Y_tr = tf.keras.utils.to_categorical(Y_tr)
        Y_val = tf.keras.utils.to_categorical(Y_val)

        out_shape = Y_tr.shape[1]
        # _______________________Build the Model___________________________________________
        model = UtilityFunctions.build_model(input_shape=parameters["input_shape"], output_shape=parameters["output_shape"], layers=parameters["layers"], dropout=parameters["dropout"], seed=parameters["seed"], neurons=parameters["neurons"])
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizers.Adam(parameters["optimizer_rate"]), metrics=[parameters["metrics"]]
        )
        # _______________________Set the callbacks___________________________________________
        callback1 = tf.keras.callbacks.EarlyStopping(monitor=parameters["monitor"], patience=parameters["patience"])
        callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=f"{path}\\model{index}.h5", save_best_only=True)

        # _____________________Train the Model_________________________________________________
        history = model.fit(X_tr, Y_tr, epochs=parameters["epochs"], batch_size=parameters["batch_size"], validation_data=(X_val, Y_val), callbacks=[callback1, callback2]).history

        # _____________________Plot accuracy and loss for the validation and Training data__________________
        PyPlots.plot_accuracy_loss(history, path, index)
        model = tf.keras.models.load_model(f"{path}\\model{index}.h5")

        # ________________Predict the activities for each of the training, validation and test data___________
        Ytrain_pr = model.predict(X_tr)
        Yval_pr = model.predict(X_val)
        Ytest_pr = model.predict(X_test)

        # evaluate and Prepare the confusion matrix for each of the training, validation and testing dataset
        for type, Ypred, Ytrue, Xtr in zip(["train", "valid", "test"], [Ytrain_pr, Yval_pr, Ytest_pr], [Y_tr, Y_val, Y_test], [X_tr, X_val, X_test]):
            # ________________________Evaluate the Model_____________________________________________
            test_loss, test_acc = model.evaluate(Xtr, Ytrue, verbose=2)
            accuracy[type] = test_acc
            loss[type] = test_loss
            ypred, ytrue = Ypred.argmax(axis=1), Ytrue.argmax(axis=1)
            precision[type] = precision_score(ypred, ytrue, average="macro")
            recall[type] = recall_score(ypred, ytrue, average="macro")
            f1[type] = f1_score(ypred, ytrue, average="macro")
            con_mat[type].append(confusion_matrix(ypred, ytrue, normalize="true"))
            mse[type].append(mean_squared_error(ypred, ytrue))

    # ______________________________Plot the confusion matrix_____________________
    meanValue = 0.0
    accuracyValue = 0.0
    PyPlots.plot_confusion_matrix(con_mat, path, labels)
    for type in ["train", "valid", "test"]:
        logger.debug(f"{type}: accuracy = {np.mean(accuracy[type])}, loss = {np.mean(loss[type])}, precision = {np.mean(precision[type])}, recall = {np.mean(recall[type])}, f1score = {np.mean(f1[type])}")
        meanValue = np.mean(mse[type])
        accuracyValue = np.mean(accuracy[type])
    return meanValue, accuracyValue

# Description: To load the data and run the MLP classifier
def load_data():
    start = time.time()
    testDF = pd.DataFrame(pd.read_csv(Constants.Dataset_test))
    Y = testDF['label']
    X = testDF.drop(['label'], axis=1)
    # X for data set; y for target set

    # Split the data into 80% test and 20% training
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=0)
    # _____________________________Parameter Declaration_________________________________________
    parameter = {
        "test_train_split_size": 0.3,
        "no_of_folds": 10,
        "input_shape": 64,
        "output_shape": 12,
        "layers": 2,
        "dropout": 0.1,
        "seed": 0,
        "neurons": 180,
        "optimizer_rate": 0.0005,
        "metrics": "accuracy",
        "monitor": "loss",
        "patience": 5,
        "epochs": 500,
        "batch_size": 32,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test
    }

    logger.debug(f"{parameter=}")
    accuracy_array: Dict[int, float] = {}
    mean_array: Dict[int, float] = {}
    accuracy_mean_array: Dict[str, Dict] = {}
    # __________________Iterating the number of layers____________________________
    for layers in range(parameter["layers"], parameter["layers"]+1):
        # _______________________ Iterating the number of neurons____________________
        for neurons in range(parameter["neurons"], parameter["neurons"]+1):
            no_of_neurons = neurons  # -----for sequential search
            # no_of_neurons = pow(2, neurons)  # -----for binary search
            # parameter["neurons"] = no_of_neurons  # -----for binary search
            mean_array[no_of_neurons], accuracy_array[no_of_neurons] = run_classifier(parameter)
        # accuracy_mean_array = {
        #     "accuracy_array": accuracy_array,
        #     "mean_array": mean_array
        # }
    logger.debug(f"{accuracy_array=} {mean_array=}")
    execution_time = time.time() - start
    logger.debug(f"{execution_time=}")
    # ______________Plot the accuracy mean square error___________________________
    # PyPlots.plot_accuracy_mean(accuracy_mean_array, path=path)

classifier=MlpClassifier()
load_data()











