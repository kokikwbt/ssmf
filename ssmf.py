""" SSMF: Shifting Seasonal Matrix Factorization """

import argparse
import warnings
from copy import deepcopy
import time
import numpy as np
from scipy.sparse import csr_array
from tqdm import trange

try:
    import ncp
    import utils
except:
    from . import ncp
    from . import utils


class SSMF:
    def __init__(self, periodicity, n_components,
                 max_regimes=100, epsilon=1e-12,
                 alpha=0.1, beta=0.05, max_iter=5, update_freq=1,
                 init_cycles=3, float_cost=32):

        assert periodicity  > 0
        assert n_components > 1
        assert max_regimes  > 0
        assert init_cycles  > 1

        self.s = periodicity
        self.k = n_components
        self.r = max_regimes
        self.g = 1  # of regimes

        self.eps = epsilon  # zero threshold
        self.alpha = alpha  # learning rate
        self.beta = beta  # A lager value may create more regimes
        self.init_cycles = init_cycles
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.float_cost = float_cost

    def initialize(self, X):
        """Initialize from a list of CSR (u, v) slices or dense (u, v, n)."""
        sparse = isinstance(X, list)
        if sparse:
            X = self._as_csr_list(X)
            d, n = X[0].shape, len(X)
        else:
            self._check_dense(X)
            d, n = X.shape[:-1], X.shape[-1]
        if n < self.init_cycles * self.s:
            raise ValueError("Input must have at least init_cycles * periodicity time points")
        self._sparse = sparse
        self.d, self.n = d, n
        
        # U(t) and V(t)
        self.U = [np.zeros((i, self.k)) for i in self.d]

        # Full history of W(t)
        self.W = np.zeros((self.r, self.s + self.n, self.k))

        # Regime history
        self.R = np.zeros(self.n, dtype=int)

        # Operation history
        self.O = np.zeros(self.n, dtype=int)
        
        # Estimate the initial factors
        if self._sparse:
            dtype = np.result_type(*[Xt.dtype for Xt in X[:self.init_cycles*self.s]])
            dtype = dtype if np.issubdtype(dtype, np.inexact) else float
            X_fold = []
            for t in range(self.s):
                Xt = X[t].astype(dtype, copy=True)
                for i in range(1, self.init_cycles):
                    Xt += X[t + i*self.s]
                Xt /= self.init_cycles
                X_fold.append(Xt)
            indices, values = self._sparse_entries(X_fold)
            tensor_indices = (
                np.concatenate([i for i, j in indices]),
                np.concatenate([j for i, j in indices]),
                np.repeat(np.arange(self.s), [Xt.nnz for Xt in X_fold]))
            factor = ncp._ncp(
                (*self.d, self.s), tensor_indices, values, self.k, maxit=3)
        else:
            X_fold = X[..., :self.init_cycles*self.s].reshape(
                *self.d, self.init_cycles, self.s).mean(axis=-2)
            factor = ncp.ncp(X_fold, self.k, maxit=3)
        self.W[:, :self.s] = factor[-1]

        # Normalization
        for i in range(len(self.d)):
            weights = np.sqrt(np.sum(factor[i] ** 2, axis=0))            
            self.U[i] = factor[i] / weights if self._sparse else factor[i] * (1 / weights)
            self.W[:, :self.s] *= weights

    @staticmethod
    def apply_grad(U, wt, Xt, alpha, eps):

        U0, U1 = U
        k = U0.shape[1]

        # The Gram terms still include the residuals at zero-valued entries.
        grad = [
            (Xt @ U1) * wt - ((U0 * wt) @ (U1.T @ U1)) * wt,
            (Xt.T @ U0) * wt - ((U1 * wt) @ (U0.T @ U0)) * wt
        ]

        wt_new = np.copy(wt)

        for i in range(2):

            # Smooth update
            grad[i] *= min(1, alpha * np.sqrt(k) / np.sqrt(np.sum(grad[i] ** 2)))
            U[i] += alpha * grad[i]

            # Normalization
            weights = np.sqrt(np.sum(U[i] ** 2, axis=0))
            # Dense reciprocal scaling retains the original rounding.
            U[i] = U[i] * (1 / weights) if isinstance(Xt, np.ndarray) else U[i] / weights
            U[i] = U[i].clip(min=eps, max=None)
            wt_new = wt_new * weights

        return U[0], U[1], wt_new

    @staticmethod
    def reconstruct(U, V, W):
        Y = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
        for t, wt in enumerate(W):
            Y[..., t] = (U * wt) @ V.T

        return Y

    @staticmethod
    def _reconstruct_at(U, V, W, indices):
        # Reconstruct stored entries one time slice at a time.
        Y = np.empty(sum(i.size for i, j in indices))
        start = 0
        for t, (i, j) in enumerate(indices):
            end = start + i.size
            Y[start:end] = np.einsum('ij,ij,j->i', U[i], V[j], W[t])
            start = end
        return Y

    @staticmethod
    def _check_dense(X, shape=None):
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            raise TypeError("Dense input must be a 3-D ndarray")
        if min(X.shape) <= 0 or (shape is not None and X.shape[:-1] != shape):
            raise ValueError("Dense input must have nonempty axes and match the initialized spatial shape")

    def _prepare_input(self, X):
        # The representation is selected once, by initialize().
        if self._sparse:
            return self._as_csr_list(X, self.d)
        self._check_dense(X, self.d)
        return X

    @staticmethod
    def _as_csr_list(X, shape=None):
        """Validate CSR input and reuse canonical slices without copying."""
        if not isinstance(X, list):
            raise TypeError("CSR input must be a list of csr_array time slices")
        if not X:
            raise ValueError("The list of CSR time slices must not be empty")
        result = X
        for t, Xt in enumerate(X):
            if not isinstance(Xt, csr_array):
                raise TypeError("Every time slice must be a csr_array")
            if shape is None:
                shape = Xt.shape
            if Xt.ndim != 2 or min(Xt.shape) <= 0 or Xt.shape != shape:
                raise ValueError("CSR time slices must have the same nonempty 2-D spatial shape")
            if not Xt.has_canonical_format:
                if result is X:
                    result = X.copy()
                result[t] = Xt.copy()
                result[t].sum_duplicates()
        return result

    @staticmethod
    def _sparse_entries(X):
        rows = np.arange(X[0].shape[0])
        indices = [(np.repeat(rows, np.diff(Xt.indptr)), Xt.indices) for Xt in X]
        values = np.concatenate([Xt.data for Xt in X])
        return indices, values

    def fit(self, X):

        X = self._prepare_input(X)
        n = len(X) if self._sparse else X.shape[-1]
        elapsed_time = np.zeros(n)

        for t in range(self.s, n):
            print('\nt=', t)

            tic = time.process_time()

            Xc = X[t-self.s:t] if self._sparse else X[..., t-self.s:t]
            self.update(Xc, t)  # Algorithm 1

            toc = time.process_time()
            elapsed_time[t] = toc - tic

        return elapsed_time

    def update(self, X, t, verbose=0):
        """ Algorithm 1 in the paper

            X: dense (u, v, s) or a list of CSR (u, v) slices for the current window
            t: current time point
        """
        # P = None  # new components
        cost1 = cost2 = np.inf
        X = self._prepare_input(X)
        entries = self._sparse_entries(X) if self._sparse else (None, X)
        self.W[:, t] = self.W[:, t - self.s]  # Copy

        cost1, ridx1 = self._regime_selection(X, t, entries)

        if t % self.update_freq == 0:
            cost2, Unew, Wnew = self._regime_generation(X, t, ridx1, self.max_iter, entries)

        if verbose > 0:
            print('RegimeSelection', cost1 + self.beta * cost1, ridx1)
            print('RegimeGeneration', cost2, self.g,
                'diff=', cost2 - (cost1 + self.beta * cost1))

        if cost1 + self.beta * cost1 < cost2:
            # print("\t---> keep")
            self.R[t] = ridx1

        else:
            # print("\t---> create")
            if self.g < self.r:
                self.R[t] = self.g
                self.U = Unew
                self.W[self.g, t - self.s + 1: t + 1] = Wnew
                self.g += 1
            else:
                self.R[t] = ridx1
                if not self.g == 1:
                    warnings.warn("# of regimes exceeded the limit")

        wt = self.W[self.R[t], t]
        Xt = X[-1] if self._sparse else X[..., -1]

        self.U[0], self.U[1], self.W[self.R[t], t] = self.apply_grad(
            self.U, wt, Xt, self.alpha, self.eps)

        # Non-negative constraint
        assert self.U[0].min() >= 0
        assert self.U[1].min() >= 0
        assert self.W.min() >= 0

    def regime_selection(self, X, t):
        X = self._prepare_input(X)
        entries = self._sparse_entries(X) if self._sparse else (None, X)
        return self._regime_selection(X, t, entries)

    def _regime_selection(self, X, t, entries):
        U, V = self.U
        n = len(X) if self._sparse else X.shape[-1]
        indices, values = entries
        E = np.zeros(self.g)

        for i in range(self.g):
            Wi = self.W[i, t - n + 1:t + 1]
            Y = (self._reconstruct_at(U, V, Wi, indices) if self._sparse
                 else self.reconstruct(U, V, Wi))
            E[i] = utils.compute_coding_cost(values, Y, self.float_cost)

        best_regime_index = np.argmin(E)
        best_coding_cost  = E[best_regime_index]

        return best_coding_cost, best_regime_index

    def regime_generation(self, X, t, ridx, max_iter=1):
        X = self._prepare_input(X)
        entries = self._sparse_entries(X) if self._sparse else (None, X)
        return self._regime_generation(X, t, ridx, max_iter, entries)

    def _regime_generation(self, X, t, ridx, max_iter, entries):
        # Initialize a new W with the nearest components
        U = deepcopy(self.U[0])
        V = deepcopy(self.U[1])
        W = self.W[ridx, t - self.s + 1:t + 1]
        
        # Fitting
        indices, values = entries
        slices = X if self._sparse else np.moveaxis(X, -1, 0)
        for _ in range(max_iter):
            for tt, Xt in enumerate(slices):
                U, V, W[tt] = self.apply_grad(
                    [U, V], W[tt], Xt, 0.5, self.eps)

        Y = (self._reconstruct_at(U, V, W, indices) if self._sparse
             else self.reconstruct(U, V, W))
        E = utils.compute_coding_cost(values, Y, self.float_cost)
        E += utils.compute_model_cost(W, self.float_cost, self.eps)

        return E, [U, V], W

    def forecast(self, ridx, current_time, forecast_time, forecast_steps=1):
        """ 
            - timepoint (int):
                A point you want to forecast
            - forecast_time (int, optional):
                length of forecast steps from the timepoint
            - forecast_steps (int, optional):
                length of forecast steps from the timepoint
        """
        U, V = self.U

        if forecast_steps == 1:
            t_seas = current_time - self.s
            t_seas += np.mod(forecast_time, self.s)
            wt = self.W[ridx, t_seas]
            # print(wt)
            return (U * wt) @ V.T

        else:
            # Forecast sequantially
            pred = [
                self.forecast(ridx, current_time, forecast_time + dt)
                for dt in range(forecast_steps)
            ]

            return np.moveaxis(pred, 0, -1)


    def fit_forecast(self, X, current_time, forecast_step=0):
        """ Perform RegimeSelection then forecasting

            X: current tensor
            current_time: current timepoint
            forecast_step:
        """
        _, ridx = self.regime_selection(X, current_time)
        return self.forecast(ridx, current_time, forecast_step)

    def test(self, X, r_test):
        """
            X: dense (u, v, n) or a list of CSR (u, v) time slices
        """
        X = self._prepare_input(X)
        n = len(X) if self._sparse else X.shape[-1]
        res = []

        for t in trange(self.s, n - r_test, desc='eval'):

            Xc = X[t-self.s+1:t+1] if self._sparse else X[..., t-self.s+1:t+1]
            self.update(Xc, t)  # Algorithm 1

            if t % r_test == 0:
                Xt = X[t:t+r_test] if self._sparse else X[..., t:t+r_test]
                times = t - self.s + np.mod(t + np.arange(r_test), self.s)
                W = self.W[self.R[t], times]
                if self._sparse:
                    indices, values = self._sparse_entries(Xt)
                    Y = self._reconstruct_at(*self.U, W, indices)
                    met = utils.eval(values, Y)
                else:
                    Y = self.reconstruct(*self.U, W)
                    met = utils.eval(Xt, Y)
                res.append(met)

        print("Total regimes=", self.g)
        print("RMSE=", np.mean(res))

    def save(self, output_dir):
        
        np.save(output_dir + '/U.npy', self.U[0])
        np.save(output_dir + '/V.npy', self.U[1])
        np.save(output_dir + '/W.npy', self.W)
        np.savetxt(output_dir + '/R.txt', self.R)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='disease')
    parser.add_argument('--output_dir', type=str, default='out')
    parser.add_argument('--periodicity', type=int, default=52)
    parser.add_argument('--n_components', type=int, default=10)
    parser.add_argument('--max_regimes', type=int, default=50)
    parser.add_argument('--max_iter', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.2)
    parser.add_argument('--penalty', type=float, default=0.05)
    parser.add_argument('--float_cost', type=int, default=32)
    parser.add_argument('--forecast_step', type=int, default=200)
    parser.add_argument('--update_freq', type=int, default=1)
    config = parser.parse_args()

    utils.make_directory(config.output_dir)

    if config.dataset == 'disease':
        tensor = utils.load_tycho(
            'data/project_tycho.csv.gz', as_tensor=True)
    
    # print(tensor.shape)

    model = SSMF(periodicity=config.periodicity,
                 n_components=config.n_components,
                 max_regimes=config.max_regimes,
                 alpha=config.learning_rate,
                 beta=config.penalty,
                 update_freq=config.update_freq,
                 float_cost=config.float_cost)

    model.initialize(tensor)
    # model.fit(tensor)  # just fit data streams and save results
    model.test(tensor, config.forecast_step)
    model.save(config.output_dir)
    utils.plot_ssmf(config.output_dir, model)

