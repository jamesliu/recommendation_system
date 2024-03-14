from __future__ import annotations
import logging

import numpy as np
from sklearn.cluster import KMeans 
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

BITS2DTYPE = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64
}

class IndexPQ:
    def __init__(
            self, 
            d:int, 
            m:int, 
            nbits:int,
            **eistimator_kwargs: str | int
    ) -> None:
        if d % m != 0:
            raise ValueError('d must be divisible by m')
        if nbits not in BITS2DTYPE:
            raise ValueError(f'nbits must be one of {list(BITS2DTYPE.keys())}')
        self.d = d
        self.m = m
        self.k = 2 ** nbits
        self.ds = d // m

        self.estimators = [
            KMeans(n_clusters=self.k, **eistimator_kwargs)
            for _ in range(self.m)
        ]
        logger.info("Created %d KMeans estimators: {self.estimators[0]!r}")
        self.is_train = False
        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32
        self.codes: np.ndarray | None = None

    def train(self, X: np.ndarray) -> None:
        if self.is_train:
            logger.warning('Already trained')
            return
        
        self.is_train = True
        for i in range(self.m):
            X_block = X[:, i * self.ds : (i + 1) * self.ds]
            self.estimators[i].fit(X_block)
        self.is_train = True

    def encode(self, X: np.ndarray) -> np.ndarray:
        if not self.is_train:
            raise ValueError('Not trained yet')
        n, d = X.shape
        if d != self.d:
            raise ValueError(f'X must have {self.d} columns')
        codes = np.zeros((n, self.m), dtype=self.dtype)
        for i in range(self.m):
            X_block = X[:, i * self.ds : (i + 1) * self.ds]
            codes[:, i] = self.estimators[i].predict(X_block)
        return codes
    
    def add(self, X: np.ndarray) -> None:
        if not self.is_train:
            raise ValueError('Not trained yet')
        codes = self.encode(X)
        self.codes = codes

    def compute_asymetric_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each row of X and each row of self.codes
        Parameters
        ----------
        X : np.ndarray of shape (n, d) of dtype np.float32
            The input data
        Returns
        -------
        np.ndarray of shape (n, n_codes) of dtype np.float32
            The distance matrix
        """
        if not self.is_train:
            raise ValueError('Not trained yet')
        if self.codes is None:
            raise ValueError('No codes, run `add` method')
        n_query = X.shape[0]
        n_codes = self.codes.shape[0]
        distance_table = np.empty((n_query, self.m, self.k), dtype=self.dtype_orig)
        for i in range(self.m):
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            centers = self.estimators[i].cluster_centers_
            distance_table[:, i, :] = euclidean_distances(
                X_i, centers, squared=True)
        distances = np.zeros((n_query, n_codes), dtype=self.dtype_orig)

        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]

        return distances
    
    def search(self, X: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the top_k nearest neighbors of each row of X in the indexed dataset
        Parameters
        ----------
        X : np.ndarray of shape (n, d) of dtype np.float32
            The input data
        top_k : int
            The number of nearest neighbors to return
        Returns
        -------
        distances
            np.ndarray of shape (n, top_k) of dtype np.float32
            The distances to the top_k nearest neighbors
        indices
            np.ndarray of shape (n, top_k) of dtype np.int32
            The indices of the top_k nearest neighbors
        """
        if not self.is_train:
            raise ValueError('Not trained yet')
        if self.codes is None:
            raise ValueError('No codes, run `add` method')
        n_queries = X.shape[0]
        distances_all = self.compute_asymetric_distance(X)
        indices =  np.argsort(distances_all, axis=1)[:, :top_k]
        distances =  np.empty((n_queries, top_k), dtype=self.dtype_orig)
        for i in range(n_queries):
            distances[i] = distances_all[i, indices[i]]
        return distances, indices

if __name__ == '__main__':
    # Create a random dataset
    np.random.seed(0)
    n = 1000
    d = 300
    m = 10
    nbits = 8
    embs = np.random.rand(n, d).astype(np.float32)
    # Instantiate and train the index
    params = {'n_init': 'auto', 'max_iter': 100}
    index = IndexPQ(d=d, m=m, nbits=nbits, **params)
    logging.basicConfig(level=logging.INFO, force=True)
    index.train(embs)
    # Add the dataset to the index
    index.add(embs)
    # Search for the 3 nearest neighbors of the first 10 elements
    distances, indices = index.search(embs[:10], top_k=3)
    print(distances)
    print(indices)
    print(index.codes[:20])

    



