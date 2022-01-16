from NN import NN
from utils import normalize, denormalize
from utils_with_external_library import save_RMSE_test


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.neural_network = NN()

        # While training comment them
        # self.WIJ = np.loadtxt("weights/WIJ.csv")
        # self.WKI = np.loadtxt("weights/WKI.csv")

    def train(self):
        self.neural_network.training()
        RMSE_test_dataset = self.neural_network.rmse_test_ds()
        print("RMSE for test data set ", RMSE_test_dataset)
        
        #saving RMSE into csv in UWEL
        save_RMSE_test(RMSE_test_dataset)

    def predict(self, input_row):
        X1_noramlized = normalize(
            valuee=float(input_row.split(",")[0]),
            minimum=self.neural_network.X1_MIN,
            maximum=self.neural_network.X1_MAX)
        X2_normalized = normalize(
            valuee=float(input_row.split(",")[1]),
            minimum=self.neural_network.X2_MIN,
            maximum=self.neural_network.X2_MAX)

        YK, _ = self.neural_network.feed_forward(WIJ=self.WIJ,
                                                 WKI=self.WKI,
                                                 X=[X1_noramlized, X2_normalized])

        Y1 = denormalize(valuee=YK[0],
                         minimum=self.neural_network.Y1_MIN,
                         maximum=self.neural_network.Y1_MAX)
        Y2 = denormalize(valuee=YK[1],
                         minimum=self.neural_network.Y2_MIN,
                         maximum=self.neural_network.Y2_MAX)

        return (Y1, Y2)

# nn = NeuralNetHolder().train()
