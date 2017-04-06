import numpy as np
from mlp.activation_functions import Sigmoid
from sklearn.metrics import accuracy_score
from mlp import NeuralNet
from mlp.util import BSSF
from rnn import util


class BPTT(NeuralNet):
    def __init__(self, features=6, hidden=40, classes=7,
                 u_back=(0, 15), u_forward=(-16, -1), v_range=(10, 30), k_back=1, k_forward=1,
                 learning_rate=0.9, a_func=Sigmoid, max_epochs=1000, patience=20,
                 validation_set=None, multi_vsets=False, classification=True):
        self.H = np.arange(hidden)
        # get correct indexes
        self._k = k_back
        self._j = k_forward
        # setup extra matrices and values
        self._hb = u_back
        self._hf = u_forward
        self._v = v_range
        # recurrent matrices
        self.V = self.input_matrix(features, *v_range)
        if u_back and k_back > 0:
            self.Ub = self.recurrent_matrix(*u_back)
            self.δb = self.delta_vecs(u_back, k_back)
        if u_forward and k_forward > 0:
            self.Uf = self.recurrent_matrix(*u_forward)
            self.δf = self.delta_vecs(u_forward, k_forward)
        super().__init__(features, hidden, classes, learning_rate, a_func, max_epochs, patience, validation_set,
                         multi_vsets, classification)
        # overwrite W so there's only one
        self.W = np.random.randn(hidden, classes)
        print("BPTT!")

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
                self._forward_prop_tt(X[i], j)
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

    # region Predict and Score
    def predict(self, X, multi_sets=False):
        out = []
        if not multi_sets:
            X = [X]
        idx = util.get_indices(X, multi_sets, self._k, self._j, shuffle=False)
        for i, j in idx:
            z = self._forward_prop_tt(X[i], j)
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
        predicted = self.predict(X, multi_sets)
        return accuracy_score(y2, predicted, sample_weight=sample_weight)
    # endregion

    def _forward_prop_tt(self, Xi, j):
        # initial activation of hidden layer
        self.Z[1] = np.ones(len(self.Z[1]))
        self.Z[1] *= .0001
        # backwards t
        if self.Ub:
            t = self._k
            while t > 0:
                x = Xi[j-t]
                xt = x.reshape(1, len(x))
                self.Z[1][slice(*self._v)] = self.activation(xt.dot(self.V) + self.b[0][slice(*self._v)])
                self.Z[1][slice(*self._hb)] = self.activation(self.Z[1][slice(*self._hb)].dot(self.Ub) +
                                                              self.b[0][slice(*self._hb)])
                t -= 1
        # t == 0
        x = Xi[j]
        x0 = x.reshape(1, len(x))
        self.Z[1][slice(*self._v)] = self.activation(x0.dot(self.V) + self.b[0][slice(*self._v)])
        # forwards t
        if self.Uf:
            t = 1
            while t <= self._j:
                x = Xi[j+t]
                xt = x.reshape(1, len(x))
                self.Z[1][slice(*self._v)] = self.activation(xt.dot(self.V) + self.b[0][slice(*self._v)])
                self.Z[1][slice(*self._hf)] = self.activation(self.Z[1][slice(*self._hf)].dot(self.Ub) +
                                                              self.b[0][slice(*self._hf)])
                t += 1
        # output layer
        self.Z[-1] = self.activation(self.Z[-2].dot(self.W) + self.b[-1])
        return self.Z[-1][0]

    def _back_prop(self, y):
        # output layer's delta: δ = (T-Z) * f'(net)
        self.δ[-1] = (y - self.Z[-1]) * self.f_prime(self.Z[-1])
        # compute deltas: δj = Σ[δk*Wjk] * f'(net)
        # for i in range(self.num_layers-2, 0, -1):
        #         self.δ[i-1] = np.tensordot(self.δ[i], self.W[i], (1, 1)) \
        #                       * self.f_prime(self.Z[i])

        # update weights: ΔWij = C*δj*Zi
        # output layer
        self.W += self.C * np.outer(self.Z[-1], self.δ[-1])
        self.b[-1] += self.C * self.δ[-1]
        # for i in range(self.num_layers-2, -1, -1):
        #     # Note since δ,W,b are all of length: num_layers-1, layer(Z[i]) == layer(b[i+1])
        #     self.W[i] += self.C * np.outer(self.Z[i], self.δ[i])
        #     self.b[i] += self.C * self.δ[i]

    def recurrent_matrix(self, start, stop):
        _len = self.H[stop] - self.H[start]
        return np.random.randn(_len, _len)

    def input_matrix(self, f, start, stop):
        _len = self.H[stop] - self.H[start]
        return np.random.randn(f, _len)

    def delta_vecs(self, h, k):
        start, stop = h
        _len = self.H[stop] - self.H[start]
        δ = []
        for i in range(k):
            δ.append(np.zeros(_len))
        return δ
