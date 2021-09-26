import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten


class UtilityFunctions:
    # Description: To reuse the code which are repetitively used
    @staticmethod
    def convert_to_numpy_array(raw_data: str) -> np.ndarray:
        return np.array(pd.read_table(raw_data, sep=" ", header=None))

    # Description: To swap the first row of the label data with the last row
    @staticmethod
    def swap_last_to_first(input_data: np.ndarray) -> np.ndarray:
        out = np.concatenate(([input_data[len(input_data)-1]], input_data[0:len(input_data)-1]))
        return out

    # Description: To build the layers of the MLP model
    def build_model(input_shape: int, output_shape: int, neurons: int, dropout: float, seed: int, layers: int) -> Model:
        model = Sequential()
        model.add(Flatten(input_shape=(input_shape, 1)))
        for i in range(layers):
            model.add(Dense(neurons))
            model.add(Activation("relu"))
        model.add(Dropout(dropout, seed=seed))
        model.add(Dense(output_shape))
        model.add(Activation("softmax"))
        return model
