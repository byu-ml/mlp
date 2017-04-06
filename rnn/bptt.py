import numpy as np
from mlp.activation_functions import Sigmoid
from sklearn.metrics import accuracy_score
from mlp import NeuralNet
from mlp.util import BSSF
from rnn import util


class BPTT(NeuralNet):
    def __init__(self, features=6, hidden=60, classes=7,
                 u_back=(0, 20), u_forward=(-21, -1), v_range=(10, 50), k_back=1, k_forward=1,
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
            self.Zb = self.Z_vecs(hidden, k_back)
            self.Zin_b = self.Z_vecs(features, k_back)
        if u_forward and k_forward > 0:
            self.Uf = self.recurrent_matrix(*u_forward)
            self.δf = self.delta_vecs(u_forward, k_forward)
            self.Zf = self.Z_vecs(hidden, k_forward)
            self.Zin_f = self.Z_vecs(features, k_forward)
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
        self.Z[1] = np.ones(self.Z[1].shape)
        self.Z[1] *= .0001
        # backwards t
        if self.Ub is not None:
            t = self._k
            for i in range(self._k):
                x = Xi[j-t+i]
                xt = x.reshape(1, len(x))
                self.Zin_b[i] = xt
                self.Z[1][:,slice(*self._v)] += self.activation(xt.dot(self.V) + self.b[0][:, slice(*self._v)])
                self.Z[1][:,slice(*self._hb)] += self.activation(self.Z[1][:,slice(*self._hb)].dot(self.Ub) +
                                                               self.b[0][:,slice(*self._hb)])
                self.Zb[i] = self.Z[1].copy()
        # t == 0
        x = Xi[j]
        self.x0 = x.reshape(1, len(x))
        self.Z[1][:,slice(*self._v)] += self.activation(self.x0.dot(self.V) + self.b[0][:,slice(*self._v)])
        # forwards t
        if self.Uf is not None:
            for i in range(self._j):
                x = Xi[j+i+1]
                xt = x.reshape(1, len(x))
                self.Z[1][:,slice(*self._v)] += self.activation(xt.dot(self.V) + self.b[0][:,slice(*self._v)])
                self.Z[1][:,slice(*self._hf)] += self.activation(self.Z[1][:,slice(*self._hf)].dot(self.Ub) +
                                                                 self.b[0][:,slice(*self._hf)])
                self.Zf[i] = self.Z[1].copy()
        # output layer
        self.Z[-1] = self.activation(self.Z[-2].dot(self.W) + self.b[-1])
        return self.Z[-1][0]

    def _back_prop(self, y):
        # output layer's delta: δ = (T-Z) * f'(net)
        self.δ[-1] = (y - self.Z[-1]) * self.f_prime(self.Z[-1])
        # compute deltas: δj = Σ[δk*Wjk] * f'(net)
        self.δ[0] = np.zeros(self.δ[0].shape)  # initially clear
        # t backwards
        if self.Ub is not None:
            self.δb[-1] = np.tensordot(self.δ[-1], self.W, (1, 1))[:,slice(*self._hb)] * self.f_prime(self.Zb[-1][:,slice(*self._hb)])
            for i in range(self._k-1, 0, -1):
                self.δb[i-1] = np.tensordot(self.δb[i], self.Ub, (1, 1)) * self.f_prime(self.Zb[i][:,slice(*self._hb)])
        # t == 0
        self.δ[0][:,slice(*self._v)] = np.tensordot(self.δ[-1], self.W, (1, 1))[:,slice(*self._v)] * self.f_prime(self.Z[1][:,slice(*self._v)])
        # t forwards
        if self.Uf is not None:
            self.δf[-1] = np.tensordot(self.δ[-1], self.W, (1, 1))[:,slice(*self._hf)] * self.f_prime(self.Zf[-1][:,slice(*self._hf)])
            for i in range(self._j-1, 0, -1):
                self.δf[i-1] = np.tensordot(self.δf[i], self.Uf, (1, 1)) * self.f_prime(self.Zf[i][:,slice(*self._hf)])

        # update weights: ΔWij = C*δj*Zi
        # output layer
        self.W += self.C * np.outer(self.Z[1], self.δ[-1])
        self.b[-1] += self.C * self.δ[-1]
        # recurrent layers
        ΔV = np.zeros(self.V.shape)
        nv = np.zeros(self.V.shape)
        Δb = np.zeros(self.b[0].shape)
        nb = np.zeros(self.b[0].shape)
        # backwards
        if self.Ub is not None:
            ΔUb = np.zeros(self.Ub.shape)
            for i in range(self._k):
                ΔUb += self.C * np.outer(self.Zb[i][:,slice(*self._hf)], self.δb[i])
                ΔV[:,slice(*self._hb)] += self.C * np.outer(self.Zin_b[i], self.δb[i])
                nv[:,slice(*self._hb)] += 1
                Δb[:,slice(*self._hb)] += self.C * self.δb[i]
                nb[:,slice(*self._hb)] += 1
            ΔUb /= self._k
            self.Ub += ΔUb
        # t == 0
        ΔV += self.C * np.outer(self.x0, self.δ[0][:,slice(*self._v)])
        nv += 1
        Δb[:,slice(*self._v)] += self.C * self.δ[0][:,slice(*self._v)]
        nb[:,slice(*self._v)] += 1
        # forwards
        if self.Uf is not None:
            ΔUf = np.zeros(self.Uf.shape)
            for i in range(self._j):
                ΔUf += self.C * np.outer(self.Zf[i][:,slice(*self._hf)], self.δf[i])
                ΔV[:,slice(*self._hf)] += self.C * np.outer(self.Zin_f[i], self.δf[i])
                nv[:,slice(*self._hf)] += 1
                Δb[:,slice(*self._hf)] += self.C * self.δf[i]
                nb[:,slice(*self._hf)] += 1
            ΔUf /= self._j
            self.Uf += ΔUf
        ΔV /= nv
        Δb = Δb / nb
        self.V += ΔV
        self.b[0] += Δb

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

    def Z_vecs(self, hidden, k):
        _Z = []
        for i in range(k):
            _Z.append(np.zeros(hidden))
        return _Z