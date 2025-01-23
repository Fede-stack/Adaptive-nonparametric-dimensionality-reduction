#AdaptiveSpectralClustering.py>

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
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.manifold import SpectralEmbedding


class AdaptiveSpectralClustering:
    def __init__(self, X, n_iter=10):
        """
        Initialize the clustering class with data matrix X.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            n_iter: Number of iterations for ID estimation
        """
        self.X = X
        self.n_iter = n_iter
        self.ids = None
        self.kstars = None
        self.similarity_matrix = None
        self.n_components = None
        self.neighs_ind = None
        
    def return_ids_kstar_binomial(self, initial_id=None, Dthr=6.67, r='opt', verbose=True):
        """
        Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.
        
        Args:
            initial_id: Initial estimate of the id (default uses 2NN)
            Dthr: Threshold value for the kstar test
            r: Parameter of binomial estimator, 0 < r < 1
            verbose: Whether to print progress
        """
        data = Data(self.X)
        
        if initial_id is None:
            data.compute_id_2NN(algorithm='base')
        else:
            data.compute_distances()
            data.set_id(initial_id)

        ids = np.zeros(self.n_iter)
        kstars = np.zeros((self.n_iter, data.N), dtype=int)

        for i in range(self.n_iter):
            data.compute_kstar(Dthr)
            if verbose:
                print("iteration ", i)
                print("id ", data.intrinsic_dim)

            r_eff = min(0.95, 0.2032**(1./data.intrinsic_dim)) if r == 'opt' else r
            rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
            rn = rk * r_eff
            n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)
            
            id = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
            data.set_id(id)

            ids[i] = id
            kstars[i] = data.kstar

        self.ids = ids
        self.kstars = kstars[(self.n_iter - 1), :]
        return self.ids, self.kstars

    def find_Kstar_neighs(self):
        """Find K* nearest neighbors for each point."""
        nn = NearestNeighbors(n_jobs=-1)
        nn.fit(self.X)

        neighs_ind = []
        for i, obs in enumerate(self.X):
            distance, ind = nn.kneighbors([obs], n_neighbors=int(self.kstars[i]) + 1)
            k_neighs = ind[0][1:]
            neighs_ind.append(k_neighs.tolist())
            
        self.neighs_ind = neighs_ind
        return self.neighs_ind

    def find_components(self):
        """Find number of components based on intrinsic dimension."""
        self.n_components = int(np.round(self.ids[self.n_iter-1]))
        return self.n_components

    def create_similarity_matrix(self, use_distances=False):
        """
        Create similarity matrix based on K* neighbors.
        
        Args:
            use_distances: If True, use euclidean distances instead of binary weights
        """
        similarity_matrix = np.zeros((self.X.shape[0], self.X.shape[0]))
        
        if use_distances:
            for i in range(similarity_matrix.shape[0]):   
                distances = [distance.euclidean(self.X[i, :], self.X[j, :]) 
                           for j in self.neighs_ind[i]]
                similarity_matrix[i, self.neighs_ind[i]] = distances
        else:
            for i in range(similarity_matrix.shape[0]):
                similarity_matrix[i, self.neighs_ind[i]] = 1.0
        
        self.similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.T)
        return self.similarity_matrix

    def return_clustering(self, n_clusters, use_n_components=True):
        """
        Perform spectral clustering using Laplacian eigenvectors.
        
        Args:
            n_clusters: Number of clusters to create
            use_n_components: If True, use estimated intrinsic dimension for embedding
        """
        if use_n_components:
            k = self.n_components
        else:
            k = n_clusters
            
        degree_matrix = np.diag(self.similarity_matrix.sum(axis=1))
        laplacian_matrix = degree_matrix - self.similarity_matrix
        eigenvalues, eigenvectors = eigh(laplacian_matrix)
        k_eigenvectors = eigenvectors[:, :k]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(k_eigenvectors)
        return kmeans.labels_

    def return_clusters(self, n_clusters):
        """
        Alternative clustering method using SpectralEmbedding.
        
        Args:
            n_clusters: Number of clusters to create
        """
        se = SpectralEmbedding(n_components=3, random_state=0, affinity='precomputed')
        embs = se.fit_transform(self.similarity_matrix)
        
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        km.fit(embs)
        return km.labels_

    def fit(self, n_clusters, use_distances=False, clustering_method=''):
        """
        Complete pipeline for clustering.
        
        Args:
            n_clusters: Number of clusters to create
            use_distances: Whether to use distances in similarity matrix
            clustering_method: 'spectral' or 'embedding' for different clustering approaches
        
        Returns:
            cluster_labels: Array of cluster assignments
        """
        #Compute IDs and k*
        self.return_ids_kstar_binomial()
        
        #Find neighbors and create similarity matrix
        self.find_Kstar_neighs()
        self.create_similarity_matrix(use_distances=use_distances)
        
        #Determine number of components
        self.find_components()
        
        if clustering_method == 'spectral':
            return self.return_clustering(n_clusters=n_clusters)
        else:
            return self.return_clusters(n_clusters=n_clusters)
