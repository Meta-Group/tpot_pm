import numpy as np
from scipy.stats import gmean, hmean
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler


class Scorer:
    """
    Scorer class: preprocesses the metrics and offers a set of scoring functions
    """

    def __init__(
        self,
        nmis: list = None,
        homos: list = None,
        comps: list = None,
        sils: list = None,
        dbs: list = None,
        chs: list = None,
        complexity: list = None,
    ):
        self._nmis = np.array(nmis) if nmis is not None else None
        self._homos = np.array(homos) if homos is not None else None
        self._comps = np.array(comps) if comps is not None else None
        self._sils = np.array(sils) if sils is not None else None
        self._dbs = np.array(dbs) if dbs is not None else None
        self._chs = np.array(chs) if chs is not None else None
        self._complexity = np.array(complexity) if complexity is not None else None
        self._population = self.preprocess_metrics()
        self._const_non_zero = 0.000001

    def preprocess_metrics(self):
        """
        Preprocesses pbarmetrics, normalizes and joins into a unique 2D array
        """
        metrics = []
        if self._nmis is not None:
            metrics.append(self._nmis)
        if self._homos is not None:
            metrics.append(self._homos)
        if self._comps is not None:
            metrics.append(self._comps)
        if self._complexity is not None:
            metrics.append(list(1-self._complexity))
        if self._sils is not None:
            metrics.append(
                list(
                    np.stack(
                        MinMaxScaler().fit_transform(
                            np.split(self._sils, self._sils.shape[0])
                        ),
                        axis=1,
                    )[0]
                )
            )
        if self._dbs is not None:
            metrics.append(
                list(
                    1
                    - np.stack(
                        MinMaxScaler().fit_transform(
                            np.split(self._dbs, self._dbs.shape[0])
                        ),
                        axis=1,
                    )[0]
                )
            )
        if self._chs is not None:
            metrics.append(
                list(
                    np.stack(
                        MinMaxScaler().fit_transform(
                            np.split(self._chs, self._chs.shape[0])
                        ),
                        axis=1,
                    )[0]
                )
            )

        return np.array(metrics).T
    
    def standardize(self, scores_list):
        return 1 - MinMaxScaler().fit_transform(scores_list.reshape(-1, 1)).reshape(1, -1)
        
    def mean_score(self):
        """
        Score based on the mean of metrics
        """
        return 1 - np.mean(self._population, axis=1)

    def median_score(self):
        """
        Score based on the median of metrics
        """
        return 1 - np.median(self._population, axis=1)

    def euclidean_score(self):
        """
        Score based on the Euclidean distance
        """

        ref = np.zeros_like(self._population[0])
        return np.array([distance.euclidean(1 - x, ref) for x in self._population])

    def seuclidean_score(self):
        """
        Score based on the standardized Euclidean distance
        Values are standardized based on metrics variance (see np.var())
        """

        ref = np.zeros_like(self._population[0])
        return np.array(
            [
                distance.seuclidean(1 - x, ref, np.var(self._population, axis=0))
                for x in self._population
            ]
        )

    def sqeuclidean_score(self):
        """
        Score based on the Euclidean distance
        """

        ref = np.zeros_like(self._population[0])
        return np.array([distance.sqeuclidean(1 - x, ref) for x in self._population])

    def minkowski_score(self):
        """
        Score based on the Minkowski distance (p equals 4)
        """

        ref = np.zeros_like(self._population[0])
        return np.array([distance.minkowski(1 - x, ref, 4) for x in self._population])

    def gmean_score(self):
        """
        Score based on the geometric mean of metrics
        """

        return 1 - gmean(self._population, axis=1)

    def hmean_score(self):
        """
        Score based on the harmonic mean of metrics
        """

        return 1 - hmean(self._population, axis=1)

    def n_max_score(self):
        """
        Score based on the number of metrics maximized
        """

        return np.array([1 / np.maximum(np.count_nonzero(x == 1), self._const_non_zero) for x in self._population])

    def div_score(self):
        """
        Score based on the division of sums
        """

        return np.array([1 / np.maximum(np.sum(x), self._const_non_zero) for x in self._population])

    def majority_score(self):
        """
        Score based on the best metric
        """

        return np.array([1 - np.amax(x) for x in self._population])