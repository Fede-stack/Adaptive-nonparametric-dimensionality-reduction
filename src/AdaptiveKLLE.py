#AdaptiveKLLE.py>

# functions

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pytest
from dadapy import Data
from dadapy._utils import utils as ut
from scipy.optimize import minimize
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.stats import ks_2samp
import sklearn
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng()

class K_starLLE:
    def __init__(self, X, initial_id, n_iter):
        self.X = X
        self.initial_id = initial_id
        self.n_iter = n_iter

    def return_ids_kstar_binomial(
            self, initial_id=None, Dthr=6.67, r='opt', verbose = True
        ):
            """Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.

            Args:
                initial_id (float): initial estimate of the id default uses 2NN
                n_iter (int): number of iteration
                Dthr (float): threshold value for the kstar test
                r (float): parameter of binomial estimator, 0 < r < 1
            Returns:
                ids, ids_err, kstars, log_likelihoods
            """
            data = Data(self.X)
            # start with an initial estimate of the ID
            if initial_id is None:
                data.compute_id_2NN(algorithm='base')
            else:
                data.compute_distances()
                data.set_id(initial_id)

            ids = np.zeros(self.n_iter)
            ids_err = np.zeros(self.n_iter)
            kstars = np.zeros((self.n_iter, data.N), dtype=int)
            log_likelihoods = np.zeros(self.n_iter)
            ks_stats = np.zeros(self.n_iter)
            p_values = np.zeros(self.n_iter)

            for i in range(self.n_iter):
                # compute kstar
                data.compute_kstar(Dthr)
                if verbose == True:
                    print("iteration ", i)
                    print("id ", data.intrinsic_dim)

                # set new ratio
                r_eff = min(0.95,0.2032**(1./data.intrinsic_dim)) if r == 'opt' else r
                # compute neighbourhoods shells from k_star
                rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
                rn = rk * r_eff
                n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)
                # compute id
                id = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
                # compute id error
                id_err = ut._compute_binomial_cramerrao(id, data.kstar-1, r_eff, data.N)
                # compute likelihood
                log_lik = ut.binomial_loglik(id, data.kstar - 1, n - 1, r_eff)
                # model validation through KS test
                n_model = rng.binomial(data.kstar-1, r_eff**id, size=len(n))
                ks, pv = ks_2samp(n-1, n_model)
                # set new id
                data.set_id(id)

                ids[i] = id
                ids_err[i] = id_err
                kstars[i] = data.kstar
                log_likelihoods[i] = log_lik
                ks_stats[i] = ks
                p_values[i] = pv

            data.intrinsic_dim = id
            data.intrinsic_dim_err = id_err
            data.intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())

            return ids, kstars[(self.n_iter - 1), :]#, ids_err, log_likelihoods, ks_stats, p_values

    def find_Kstar_neighs(self, kstars):
        nn = NearestNeighbors(n_jobs=-1)
        nn.fit(self.X)

        neighs_ind = []
        for i, obs in enumerate(self.X):
            distance, ind = nn.kneighbors([obs], n_neighbors=kstars[i] + 1)

            k_neighs = ind[0][1:]
            neighs_ind.append(k_neighs.tolist())
        return neighs_ind

    def find_components(self, ids):
        n_components = int(np.round(ids[self.n_iter-1]))
        return n_components

    def K_star_local_linear_embedding(self, indices, n_components):
        W = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in range(self.X.shape[0]):
            n_neighbors = len(indices[i])
            Z = self.X[indices[i]] - self.X[i]  
            C = np.dot(Z, Z.T)  
            trace = np.trace(C)
            if trace > 0:
                R = 1e-3 * trace
            else:
                R = 1e-3
            C.flat[:: n_neighbors + 1] += R
            w = solve(C, np.ones(n_neighbors), assume_a="pos")  
            W[i, indices[i]] = w / np.sum(w)

        M = (np.eye(self.X.shape[0]) - W).T @ (np.eye(self.X.shape[0]) - W)
        eigenvalues, eigenvectors = eigh(M)
        Y = eigenvectors[:, 1:n_components+1]

        return Y, W
    
    def calculate_embedding(self, initial_id=None, Dthr=23.92812698, r='opt'):
        ids, kstars = self.return_ids_kstar_binomial(initial_id=initial_id, Dthr=Dthr, r=r)
        indices = self.find_Kstar_neighs(kstars)
        n_components = self.find_components(ids)
        Y, W = self.K_star_local_linear_embedding(indices, n_components)
        return Y, W, kstars

    def calculate_n_dimembedding(self, initial_id=None, Dthr=6.67, r='opt', n_comps = 2):
        ids, kstars = self.return_ids_kstar_binomial(initial_id=initial_id, Dthr=Dthr, r=r)
        indices = self.find_Kstar_neighs(kstars)
        Y, W = self.K_star_local_linear_embedding(indices, n_comps)
        return Y, W, kstars
    
    def calculate_reconstruction_error(self, Y, W):
        return np.sum((Y - np.dot(W, Y))**2)
