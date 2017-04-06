import numpy as np
from mlp.activation_functions import Sigmoid
from sklearn.metrics import accuracy_score
from mlp import NeuralNet
from mlp.util import BSSF
from rnn import util


class WindowedMLP(NeuralNet):
    def __init__(self, features, hidden=20, classes=2, window=(-1), learning_rate=0.9, a_func=Sigmoid, max_epochs=1000, patience=20,
                 validation_set=None, multi_vsets=False, classification=True):
        # window params
        w = np.array(window)
        self._k = 0
        if len(w[w < 0]) > 0:
            self._k = max(abs(w[w < 0]))
        self._j = 0
        if len(w[w > 0]) > 0:
            self._j = max(w[w > 0])
        self._window = np.zeros(len(w)+1, dtype=int)
        self._window[1:] = w
        # adjust input size
        n_features = features * len(self._window)
        print("window:", self._window)
        print("# features:", n_features)
        super().__init__(n_features, hidden, classes, learning_rate, a_func, max_epochs, patience, validation_set,
                         multi_vsets, classification)

    def fit(self, X, Y, multi_sets=False):
        epoch = 0
        Δp = 0
        bssf = BSSF(self.W, self.b, 0)
        if not multi_sets:
            X = [X]
            Y = [Y]
        while epoch < self._max_epochs and Δp < self._patience:
            idx = util.get_indices(X, multi_sets, self._k, self._j)
            for i, j in idx:
                self._forward_prop(self.get_window(X, i, j))
                self._back_prop(Y[i][j])
            epoch += 1
            # Do validation check
            if self._VS:
                score = self.score(self._VS[0], self._VS[1], multi_sets=self._multi_vsets)
                if score > bssf.score:
                    bssf = BSSF(self.W, self.b, score)
                    Δp = 0
                else:
                    Δp += 1
        # if training stopped because of patience, use bssf instead
        if self._VS and Δp >= self._patience:
            self.W = bssf.W
            self.b = bssf.b
        return epoch

    def predict(self, X, multi_sets=False):
        out = []
        if not multi_sets:
            X = [X]
        idx = util.get_indices(X, multi_sets, self._k, self._j, shuffle=False)
        for i, j in idx:
            x = self.get_window(X, i, j)
            z = self._forward_prop(x)
            if self._classification:
                q = np.zeros(z.shape)
                q[z.argmax()] = 1.
                out.append(q)
            else:
                out.append(z)
        return np.array(out)

    def score(self, X, y, sample_weight=None, multi_sets=False):
        y2 = []
        if not multi_sets:
            y = [y]
        idx = util.get_indices(y, multi_sets, self._k, self._j, shuffle=False)
        for i, j in idx:
            y2.append(y[i][j])
        y2 = np.array(y2)
        # if multi_sets:
        #     y2 = y[0]
        #     for yi in y[1:]:
        #         y2 = np.vstack((y2, yi))
        # else:
        #     y2 = y
        predicted = self.predict(X, multi_sets)
        return accuracy_score(y2, predicted, sample_weight=sample_weight)

    def get_window(self, X, i, j):
        return X[i][self._window + j].flatten()
