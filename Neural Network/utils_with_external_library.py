import datetime

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import normalize


def pre_processing(nn_obj, train_size, test_val_size, random_state):
    dataset = pd.read_csv('game_datset.csv')

    minimum = dataset.min()
    maximum = dataset.max()

    nn_obj.X1_MIN = minimum[0]
    nn_obj.X1_MAX = maximum[0]
    nn_obj.X2_MIN = minimum[1]
    nn_obj.X2_MAX = maximum[1]
    nn_obj.Y1_MIN = minimum[2]
    nn_obj.Y1_MAX = maximum[2]
    nn_obj.Y2_MIN = minimum[3]
    nn_obj.Y2_MAX = maximum[3]

    normalized_DS = normalize(valuee=dataset, minimum=minimum, maximum=maximum)

    input_DS = normalized_DS.iloc[:, 0:2]
    output_DS = normalized_DS.iloc[:, 2:4]

    # spliting data into train and (test,val)
    input_train_DS, input_test_val_DS, output_train_DS, output_test_val_DS = train_test_split(
        input_DS, output_DS, train_size=train_size, random_state=random_state)

    # spliting data into test and val from (test,val)
    input_test_DS, input_val_DS, output_test_DS, output_val_DS = train_test_split(
        input_test_val_DS, output_test_val_DS, train_size=test_val_size, random_state=random_state)

    nn_obj.X_training_dataset = input_train_DS.values
    nn_obj.Y_training_dataset = output_train_DS.values
    nn_obj.X_validation_dataset = input_val_DS.values
    nn_obj.Y_validation_dataset = output_val_DS.values
    nn_obj.X_testing_dataset = input_test_DS.values
    nn_obj.Y_testing_dataset = output_test_DS.values

    nn_obj.input_number = input_DS.shape[1]
    nn_obj.output_number = output_DS.shape[1]


def plot_rmse(epoch_rmse_validation_ds, epoch_rmse_training_ds, epoch):
    plot.plot(range(epoch), epoch_rmse_validation_ds, label="validation")
    plot.plot(range(epoch), epoch_rmse_training_ds, label="training")
    plot.xlabel('epochs')
    plot.ylabel('validation\ntraining')
    plot.title('Training and Validation Dataset RMSE')
    plot.legend()
    plot.show()


def weights_save(WIJ, WKI):
    d = datetime.datetime.now()
    np.savetxt("weights/WIJ.csv", WIJ)
    np.savetxt("weights/WKI.csv", WKI)

#saving the RMSE of test dataset in csv file from NNH
def save_RMSE_test(RMSE_test):
    file_name = "weights/rmse"
    with open(f"{file_name}.txt", "w") as text_file:
        text_file.write(f"Root Mean Square Error (RMSE) of the test dataset is {RMSE_test}")
