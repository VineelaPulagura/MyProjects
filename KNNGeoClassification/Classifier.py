# Created By    : Vineela Pulagura
# Created on    : 27-06-2021
import logging
import os
import sys
import re
from builtins import str

import numpy as np


from datetime import datetime
from logging import getLogger, StreamHandler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.tests.test_base import K
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interpn
from typing import Any

# ________Setting a path to save the log files and figures_________
from tensorflow.python.ops.metrics_impl import metric_variable

path = "Logs\\KNN_results_"+datetime.now().strftime("%Y%m%d_%H%M%S")
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

class KNNClassifier:

    def run_classifier(self):

        mean_fpr_strt = []
        mean_tpr_strt = []
        mean_auc_strt = []
        j = 1
        Accuracy = {"WideStreet": [], "MiddleWideStreet": [], "NarrowStreet": []}
        # _____________________Iterate the streets_____________________________________________________
        for streetName in ["WideStreet", "MiddleWideStreet", "NarrowStreet"]:
            X_train, Y_train = np.array([]), np.array([])
        # _____________________Iterate the Street sites________________________________________________
            for streetSite in ["LeftSide", "RightSide"]:
                InputFile = 'Dataset\\StreetData\\' + streetName + '\\' + streetSite + '\\' + streetSite + '.gpx'
                data = open(InputFile).read()
                lat = np.array(re.findall(r'lat="([^"]+)', data), dtype=float)
                lon = np.array(re.findall(r'lon="([^"]+)', data), dtype=float)

                X_values = np.array(list(zip(lat, lon)))
                print(X_values.shape)
        # _____________________label the street site based on the type of streetSite___________________
                Y_values = np.vstack(np.zeros(shape=X_values.shape[0])) if streetSite == "LeftSide" else np.vstack(np.ones(shape=X_values.shape[0]))
                Y_values = Y_values.tolist()

                X_train = np.append(X_train, X_values)
                Y_train = np.append(Y_train, Y_values)

            y = int(X_train.shape[0]/2)
            print(y)
            X_train = np.reshape(X_train, (y, 2))
            X_train, Y_train = shuffle(X_train, Y_train)
            print(X_train.shape)
            print(Y_train.shape)
            logger.debug(f"{X_train.shape=}, {Y_train.shape=}")

        # _____________________List Hyperparameters  to tune______________________________________
            n_neighbors = list(range(1, 20))
            p = [1, 2, 3]

        # _____________________Convert to dictionary_____________________________________________
            hyperparameters = dict(n_neighbors=n_neighbors, p=p)

        # _____________________Create a KNN object_______________________________________________
            knn_2 = KNeighborsClassifier()

        # _____________________Use GridSearch____________________________________________________
            gridSearch = GridSearchCV(knn_2, hyperparameters, cv=5)

        # _____________________Fit the model_____________________________________________________
            model = gridSearch.fit(X_train, Y_train)

        # _____________________Print The value of best Hyperparameters___________________________
            logger.debug(f"{model.best_estimator_.get_params()['p']=}")
            logger.debug(f"{model.best_estimator_.get_params()['n_neighbors']=}")

        # _____________________Iterate the model to visualize the hyper parameters___________________________
            accuracy_mean_p1 = []
            accuracy_mean_p2 = []
            accuracy_mean_p3 = []
            k_range = n_neighbors
            for k in k_range:
                knn_p1 = KNeighborsClassifier(n_neighbors=k, p=1)
                knn_p2 = KNeighborsClassifier(n_neighbors=k)
                knn_p3 = KNeighborsClassifier(n_neighbors=k, p=3)

                accuracy_mean_p1.append(np.mean(cross_val_score(knn_p1, X_train, Y_train, cv=10, scoring='accuracy')))
                accuracy_mean_p2.append(np.mean(cross_val_score(knn_p2, X_train, Y_train, cv=10, scoring='accuracy')))
                accuracy_mean_p3.append(np.mean(cross_val_score(knn_p3, X_train, Y_train, cv=10, scoring='accuracy')))

            fig = plt.figure(figsize=(20, 10))
            plt.plot(k_range, accuracy_mean_p1, label='Accuracy_Manhattan')
            plt.plot(k_range, accuracy_mean_p2, label='Accuracy_Euclidean')
            plt.plot(k_range, accuracy_mean_p3, label='Accuracy_Minkowski')
            plt.xticks(k_range, fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('K', fontsize=18)
            plt.ylabel('Accuracy', fontsize=18)
            plt.title(streetName, fontsize=18)
            plt.legend(fontsize=18)
            fig.savefig(f"{path}\\{streetName}.png")

            # _____________________use cross fold validation to visualize each fold with accuracy___________________________
            cv = StratifiedKFold(n_splits=10)
            fig = plt.figure(figsize=[12,12])
            ax1 = fig.add_subplot(111,aspect = 'equal')
            ax1.add_patch(
                patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)
                )
            ax1.add_patch(
                patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)
                )

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            i = 1
            knn = KNeighborsClassifier(n_neighbors=model.best_estimator_.get_params()['n_neighbors'], p=model.best_estimator_.get_params()['p'])
            for train,test in cv.split(X_train,Y_train):
                prediction = knn.fit(X_train[train],Y_train[train]).predict_proba(X_train[test])
                fpr, tpr, t = roc_curve(Y_train[test], prediction[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
                i= i+1
            # _____________________Plot the mean ROC curves of all the folds___________________________
            plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color='blue',
                     label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC-{streetName}")
            plt.legend(loc="lower right")
            plt.text(0.32,0.7,'More accurate area',fontsize = 12)
            plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
            fig.savefig(f"{path}\\ROC-{streetName}.png")
            plt.show()
            mean_fpr_strt = mean_fpr
            mean_tpr_strt.append(mean_tpr)
            mean_auc_strt.append(mean_auc)

        # _____________________plot the ROC curve for all the three streets___________________________
        fig1 = plt.figure(figsize=[12, 12])
        ax = fig1.add_subplot(111, aspect='equal')
        ax.add_patch(
            patches.Arrow(0.45, 0.5, -0.25, 0.25, width=0.3, color='green', alpha=0.5)
        )
        ax.add_patch(
            patches.Arrow(0.5, 0.45, 0.25, -0.25, width=0.3, color='red', alpha=0.5)
        )
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        plt.plot(mean_fpr_strt, mean_tpr_strt[0], color='blue' ,lw=2, alpha=1,
                 label=f"ROC- Wide Street (AUC = %0.2f)" % (mean_auc_strt[0]))
        plt.plot(mean_fpr_strt, mean_tpr_strt[1], color='green', lw=2, alpha=1,
                 label=f"ROC Middle Wide Street (AUC = %0.2f)" % (mean_auc_strt[1]))
        plt.plot(mean_fpr_strt, mean_tpr_strt[2], color='red', lw=2, alpha=1,
                 label=f"ROC Narrow Street (AUC = %0.2f)" % (mean_auc_strt[2]))


        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title(f"ROC-Streets", fontsize=20)
        plt.legend(loc="lower right")
        plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
        plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)
        fig1.savefig(f"{path}\\ROC-Streets.png")

        plt.show()

if __name__ == '__main__':
    classifier = KNNClassifier()
    classifier.run_classifier()

