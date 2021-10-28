""" SSMF: Shifting Seasonal Matrix Factorization """

import argparse
import warnings
from copy import deepcopy
import time
import numpy as np
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

        self.d = X.shape[:-1]
        self.n = X.shape[-1]
        
        # U(t) and V(t)
        self.U = [np.zeros((i, self.k)) for i in self.d]

        # Full history of W(t)
        self.W = np.zeros((self.r, self.s + self.n, self.k))

        # Regime history
        self.R = np.zeros(self.n, dtype=int)

        # Operation history
        self.O = np.zeros(self.n, dtype=int)
        
        # Estimate the initial factors
        X_fold = [X[..., i*self.s:(i+1)*self.s] for i in range(self.init_cycles)]
        X_fold = np.array(X_fold).sum(axis=0) / self.init_cycles
        factor = ncp.ncp(X_fold, self.k, maxit=3)
        self.W[:, :self.s] = factor[-1]

        # Normalization
        for i in range(len(self.d)):
            weights = np.sqrt(np.sum(factor[i] ** 2, axis=0))            
            self.U[i] = factor[i] @ np.diag(1 / weights)
            self.W[:, :self.s] = self.W[:, :self.s] @ np.diag(weights)

    @staticmethod
    def apply_grad(U, wt, Xt, alpha, eps):

        U0, U1 = U
        D = np.diag(wt)
        k = U0.shape[1]

        grad = [
            Xt @ U1 @ D - U0 @ D @ (U1.T @ U1) @ D,
            Xt.T @ U0 @ D - U1 @ D @ (U0.T @ U0) @ D
        ]

        wt_new = np.copy(wt)

        for i in range(2):

            # Smooth update
            grad[i] *= min(1, alpha * np.sqrt(k) / np.sqrt(np.sum(grad[i] ** 2)))
            U[i] += alpha * grad[i]

            # Normalization
            weights = np.sqrt(np.sum(U[i] ** 2, axis=0))
            U[i] = U[i] @ np.diag((1 / weights))
            U[i] = U[i].clip(min=eps, max=None)
            wt_new = wt_new * weights

        return U[0], U[1], wt_new

    @staticmethod
    def reconstruct(U, V, W):
        Y = np.zeros((U.shape[0], V.shape[0], W.shape[0]))
        for t, wt in enumerate(W):
            Y[..., t] = U @ np.diag(wt) @ V.T

        return Y

    def fit(self, X):

        n = X.shape[-1]
        elapsed_time = np.zeros(n)

        for t in range(self.s, n):
            print('\nt=', t)

            tic = time.process_time()

            Xc = X[..., t-self.s:t]
            self.update(Xc, t)  # Algorithm 1

            toc = time.process_time()
            elapsed_time[t] = toc - tic

        return elapsed_time

    def update(self, X, t, verbose=0):
        """ Algorithm 1 in the paper

            X: current tensor (u, v, s)
            t: current time point
        """
        # P = None  # new components
        cost1 = cost2 = np.inf
        self.W[:, t] = self.W[:, t - self.s]  # Copy

        cost1, ridx1 = self.regime_selection(X, t)

        if t % self.update_freq == 0:
            cost2, Unew, Wnew = self.regime_generation(X, t, ridx1, self.max_iter)

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
        Xt = X[..., -1]

        self.U[0], self.U[1], self.W[self.R[t], t] = self.apply_grad(
            self.U, wt, Xt, self.alpha, self.eps)

        # Non-negative constraint
        assert self.U[0].min() >= 0
        assert self.U[1].min() >= 0
        assert self.W.min() >= 0

    def regime_selection(self, X, t):
        
        U, V = self.U
        n = X.shape[-1]
        Y = np.zeros(X.shape)
        E = np.zeros(self.g)

        for i in range(self.g):
            Wi = self.W[i, t - n + 1:t + 1]
            Y = self.reconstruct(U, V, Wi)
            E[i] = utils.compute_coding_cost(X, Y, self.float_cost)

        best_regime_index = np.argmin(E)
        best_coding_cost  = E[best_regime_index]

        return best_coding_cost, best_regime_index

    def regime_generation(self, X, t, ridx, max_iter=1):
        
        # Initialize a new W with the nearest components
        n = X.shape[-1]
        U = deepcopy(self.U[0])
        V = deepcopy(self.U[1])
        W = np.zeros((self.s, self.k))
        W = self.W[ridx, t - self.s + 1:t + 1]
        
        # Fitting
        for _ in range(max_iter):
            for tt in range(n):
                U, V, W[tt] = self.apply_grad(
                    [U, V], W[tt], X[..., tt], 0.5, self.eps)

        Y = self.reconstruct(U, V, W)
        E = utils.compute_coding_cost(X, Y, self.float_cost)
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
            return U @ np.diag(wt) @ V.T

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
            X: a tensor
        """
        n = X.shape[-1]
        Y = np.zeros(X.shape)
        res = []

        for t in trange(self.s, n - r_test, desc='eval'):

            Xc = X[..., t-self.s:t]
            self.update(Xc, t)  # Algorithm 1

            if t % r_test == 0:
                Y[..., t:t+r_test] = self.forecast(
                    self.R[t], t, t, forecast_steps=r_test)
                met = utils.eval(X[..., t:t+r_test], Y[..., t:t+r_test])
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

