# Created By    : Vineela Pulagura
# Created on     : 25-01-2021
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List
import seaborn as sns


class PyPlots:
    # Description: To plot the accuracy and loss of validation and Training data for each fold or split
    @staticmethod
    def plot_accuracy_loss(model_fit: Any, path: str, index: int):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
        ax1.plot(model_fit['accuracy'], label="train")
        ax1.plot(model_fit['val_accuracy'], label="validation")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy')
        ax1.legend()

        ax2.plot(model_fit['loss'], label="train")
        ax2.plot(model_fit['val_loss'], label="validation")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss')
        ax2.legend()

        fig.savefig(f"{path}\\figure{index}.png")

    # Description: To plot the confusion matrix for each of the Training, Validation and Test data
    @classmethod
    def plot_confusion_matrix(cls, con_mat, path, labels):
        fig, ax = plt.subplots(ncols=3, figsize=(35, 10))
        for i, cat in enumerate(["train", "valid", "test"]):
            cm = np.mean(con_mat[cat], axis=0)
            sns.heatmap(
                cm,
                annot=True,
                cmap="YlGnBu",
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax[i],
            )
            ax[i].set_xlabel("Predicted label")
            ax[i].set_ylabel("True label")
            ax[i].set_title(f"Confusion matrix - {cat}")
        plt.tight_layout()
        fig.savefig(f"{path}\\conmat.png", dpi=300)
        plt.close()

    # Description: To plot the accuracy and mse for the different hyperparameters
    @staticmethod
    def plot_accuracy_mean(accuracy_mean_array: Any, path: str):
        fig = plt.figure()
        plt.plot(accuracy_mean_array["accuracy_array"].keys(), accuracy_mean_array["accuracy_array"].values(), label="accuracy")
        plt.plot(accuracy_mean_array["mean_array"].keys(), accuracy_mean_array["mean_array"].values(), label="mean square error")
        plt.xlabel("Number of neurons")
        plt.title("Accuracy")
        plt.legend()
        plt.show()
        fig.savefig(f"{path}\\accuracyMeanSquareErrorPlot.png")
        plt.close()
