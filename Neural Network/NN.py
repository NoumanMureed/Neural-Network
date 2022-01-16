from constants import *
from utils import *
from utils_with_external_library import *


class NN:
    def __init__(self):
        self.train_size = TRAIN_RATIO
        self.validation_size = (1 - self.train_size) / 2
        self.test_size = (1 - self.train_size) / 2
        self.test_validation_split = 0.5
        self.random_state = RANDOM_STATE
        self.lambdaa = LAMBDA
        self.learning_rate = LEARNING_RATE
        self.alpha = ALPHA
        self.epoch = EPOCH
        self.stopping_criteria_count = STOPPING_CRITERIA_MAXIMUM_COUNT
        self.stopping_criteria_difference = STOPPING_CRITERIA_DIFFERENCE
        self.input_number = None
        self.output_number = None
        self.hidden_neuron_number = HIDDEN_NEURON_NUMBER

        self.X1_MIN = None
        self.X1_MAX = None
        self.X2_MIN = None
        self.X2_MAX = None
        self.Y1_MIN = None
        self.Y1_MAX = None
        self.Y2_MIN = None
        self.Y2_MAX = None

        self.X_training_dataset = None
        self.Y_training_dataset = None
        self.X_validation_dataset = None
        self.Y_validation_dataset = None
        self.X_testing_dataset = None
        self.Y_testing_dataset = None

        pre_processing(
            self, self.train_size, self.test_validation_split, self.random_state
        )

        self.WIJ = random_matrix(self.hidden_neuron_number, self.input_number)
        self.WKI = random_matrix(self.output_number, self.hidden_neuron_number)

        self.previous_delta_WKI = zero_matrix(self.output_number, self.hidden_neuron_number)
        self.previous_delta_WIJ = zero_matrix(self.hidden_neuron_number, self.input_number)

        self.best_WIJ = None
        self.best_WKI = None
        self.training_EK_DS = []

        self.epoch_rmse_training_ds = []
        self.epoch_rmse_validation_ds = []
        self.best_epoch = 0

    def feed_forward(self, WIJ, WKI, X):
        YK = []
        VK = []
        for k in range(self.output_number):
            HI = []
            VI = []
            sum_at_k = 0
            for i in range(self.hidden_neuron_number):
                sum_at_i = 0
                for j in range(self.input_number):
                    sum_at_i += WIJ[i][j] * X[j]
                VI.append(sum_at_i)
                HI.append(sigmoid(sum_at_i, self.lambdaa))

                sum_at_k += WKI[k][i] * HI[i]
            VK.append(sum_at_k)
            YK.append(sigmoid(sum_at_k, self.lambdaa))
        return YK, HI

    def error_k(self, Y, YK):
        EK = []
        for k in range(self.output_number):
            EK.append(Y[k] - YK[k])
        return EK

    def local_gradient_k_i(self, YK, EK, HI, WKI):
        local_gradient_i = []
        for i in range(self.hidden_neuron_number):
            local_gradients_k = []
            local_graident_k_WKI = 0
            for k in range(self.output_number):
                local_gradient_k = self.lambdaa * YK[k] * (1 - YK[k]) * EK[k]
                local_gradients_k.append(local_gradient_k)
                local_graident_k_WKI += local_gradient_k * WKI[k][i]
            local_gradient_i.append(
                self.lambdaa * HI[i] * (1 - HI[i]) * local_graident_k_WKI
            )

        return local_gradients_k, local_gradient_i

    def delta_WKI(self, LGK, HI, previous_delta_WKI):
        delta_WKI_matrix = []
        for k in range(self.output_number):
            delta_WKI_column = []
            for i in range(self.hidden_neuron_number):
                delta_WKI_column.append(
                    (self.learning_rate * LGK[k] * HI[i])
                    + (self.alpha * previous_delta_WKI[k][i])
                )
            delta_WKI_matrix.append(delta_WKI_column)
        return delta_WKI_matrix

    def delta_WIJ(self, LGI, XJ, previous_delta_WIJ):
        delta_WIJ_matrix = []
        for i in range(self.hidden_neuron_number):
            delta_WIJ_column = []
            for j in range(self.input_number):
                delta_WIJ_column.append(
                    (self.learning_rate * LGI[i] * XJ[j])
                    + (self.alpha * previous_delta_WIJ[i][j])
                )
            delta_WIJ_matrix.append(delta_WIJ_column)
        return delta_WIJ_matrix

    def updating_weights(self):
        for XJ, Y in zip(self.X_training_dataset, self.Y_training_dataset):
            YK, HI = self.feed_forward(self.WIJ, self.WKI, XJ)
            EK = self.error_k(Y, YK)
            self.training_EK_DS.append(EK)
            LGK, LGI = self.local_gradient_k_i(YK, EK, HI, self.WKI)
            delta_WKI = self.delta_WKI(LGK, HI, self.previous_delta_WKI)
            delta_WIJ = self.delta_WIJ(LGI, XJ, self.previous_delta_WIJ)

            self.previous_delta_WKI = delta_WKI
            self.previous_delta_WIJ = delta_WIJ

            # Weight updates
            for k in range(self.output_number):
                for i in range(self.hidden_neuron_number):
                    self.WKI[k][i] = self.WKI[k][i] + delta_WKI[k][i]

            for i in range(self.hidden_neuron_number):
                for j in range(self.input_number):
                    self.WIJ[i][j] = self.WIJ[i][j] + delta_WIJ[i][j]

    def rmse_validation_ds(self):
        validation_dataset_EK = []
        for X, Y in zip(self.X_validation_dataset, self.Y_validation_dataset):
            YK, _ = self.feed_forward(WIJ=self.WIJ, WKI=self.WKI, X=X)
            EK = self.error_k(Y=Y, YK=YK)
            validation_dataset_EK.append(EK)
        return root_mean_square_error(validation_dataset_EK)

    def rmse_test_ds(self):
        test_dataset_EK = []
        for X, Y in zip(self.X_testing_dataset, self.Y_testing_dataset):
            YK, _ = self.feed_forward(WIJ=self.WIJ, WKI=self.WKI, X=X)
            EK = self.error_k(Y=Y, YK=YK)
            test_dataset_EK.append(EK)
        return root_mean_square_error(test_dataset_EK)

    def early_stopping(self, current_rmse, current_count, previous_rmse, epoch):
        difference_rmse_val = previous_rmse - current_rmse
        if epoch == 0:
            difference_rmse_val = abs(difference_rmse_val)
        if difference_rmse_val < self.stopping_criteria_difference:
            current_count += 1
            if current_count == 1:
                self.best_epoch = epoch

        if difference_rmse_val > self.stopping_criteria_difference:
            current_count = 0
        if current_count == self.stopping_criteria_count:
            return True, current_count, difference_rmse_val
        return False, current_count, difference_rmse_val
        

    def training(self):

        WIJ_list = []
        WKI_list = []
        e = 0
        current_count = 0
        previous_rmse = 0
        for e in range(self.epoch):
            e += 1

            # training data error
            self.training_EK_DS = []
            # calling update weight function
            self.updating_weights()

            # appending weights into WIJ list and WKI list
            WIJ_list.append(self.WIJ)
            WKI_list.append(self.WKI)

            training_DS_RMSE = root_mean_square_error(self.training_EK_DS)
            validation_DS_RMSE = self.rmse_validation_ds()
            print("epoch ------ ", e)
            print(f"validation:  {validation_DS_RMSE},    training: {training_DS_RMSE}")

            # plot graph of train and validation error
            self.epoch_rmse_training_ds.append(training_DS_RMSE)
            self.epoch_rmse_validation_ds.append(validation_DS_RMSE)

            early_stop, current_count, _ = self.early_stopping(
                current_rmse=validation_DS_RMSE,
                current_count=current_count,
                previous_rmse=previous_rmse,
                epoch=e,
            )
            # getting the best epoch weight
            if early_stop:
                self.best_WIJ = WIJ_list[self.best_epoch]
                self.best_WKI = WKI_list[self.best_epoch]
                # saving best weights into csv in UWEL class
                weights_save(self.best_WIJ, self.best_WKI)
                plot_rmse(self.epoch_rmse_validation_ds, self.epoch_rmse_training_ds, e)

                return 
            previous_rmse = validation_DS_RMSE

        #self.best_WIJ = self.WIJ
        #self.best_WKI = self.WKI
        #weights_save(self.WIJ, self.WKI)
        #plot_rmse(self.epoch_rmse_validation_ds, self.epoch_rmse_training_ds, e)
