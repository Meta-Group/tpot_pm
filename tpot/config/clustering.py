# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

clustering_config_dict = {

    # Clusterers
    
    'sklearn.cluster.AgglomerativeClustering': {
        'n_clusters': range(2, 105),
        'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed', 'cityblock'],
        'linkage': ['ward', 'complete', 'average', 'single'],
    },

    # 'sklearn.cluster.Birch': {
    #     'threshold': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
    #     'branching_factor': [2, 5, 10, 25, 50, 100],
    #     'n_clusters': range(1, 105),
    # },

    'sklearn.cluster.DBSCAN': {
        'eps': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'min_samples': [1, 3, 5, 10, 25, 50],
        'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed', 'cityblock'],
        'leaf_size': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    },

    # 'sklearn.cluster.KMeans': {
    #     'n_clusters': range(1, 105),
    #     'init': ['k-means++', 'random'],
    #     'algorithm': ['lloyd', 'elkan'],
    #     'max_iter': np.arange(100, 600, 100)
    # },

    # 'sklearn.cluster.BisectingKMeans': {
    #     'n_clusters': range(1, 105),
    #     'init': ['k-means++', 'random'],
    #     'algorithm': ['lloyd', 'elkan'],
    #     'max_iter': np.arange(100, 600, 100),
    #     'bisecting_strategy': ['biggest_inertia', 'largest_cluster'],
    # },

    'sklearn.cluster.MiniBatchKMeans': {
        'n_clusters': range(2, 105),
        'init': ['k-means++', 'random'],
        'max_iter': np.arange(100, 600, 100),
    },

    # 'sklearn.cluster.MeanShift': {
    #     'max_iter': np.arange(100, 600, 100),
    #     'min_bin_freq': [1, 3, 5, 10],
    #     'cluster_all': [True, False],
    # },

    # 'sklearn.cluster.OPTICS': {
    #     'min_samples': [2, 3, 5, 10, 25, 50],
    #     'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed', 'cityblock'],
    #     'cluster_method': ['xi', 'dbscan'],
    #     'leaf_size': [1,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # },

    'sklearn.cluster.SpectralClustering': {
        'n_clusters': range(2, 105),
        'eigen_solver': ['arpack', 'lobpcg', 'amg'],
        'affinity': ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
    },

    # Preprocessors
    #'sklearn.preprocessing.Binarizer': {
     #   'threshold': np.arange(0.0, 1.01, 0.05)
    #},

    # 'sklearn.cluster.FeatureAgglomeration': {
    #     'linkage': ['ward', 'complete', 'average'],
    # },

#     'sklearn.preprocessing.MaxAbsScaler': {
#     },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2']
    },

#     'sklearn.kernel_approximation.Nystroem': {
#         'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
#         'gamma': np.arange(0.0, 1.01, 0.05),
#         'n_components': range(1, 11)
#     },

#     'sklearn.preprocessing.PolynomialFeatures': {
#         'degree': [2],
#         'include_bias': [False],
#         'interaction_only': [False]
#     },

#     'sklearn.kernel_approximation.RBFSampler': {
#         'gamma': np.arange(0.0, 1.01, 0.05)
#     },

#     'sklearn.preprocessing.RobustScaler': {
#     },

    'sklearn.preprocessing.StandardScaler': {
    },

#     'tpot.builtins.ZeroCount': {
#     },

#     'tpot.builtins.OneHotEncoder': {
#         'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
#         'sparse': [False],
#         'threshold': [10]
#     },


    # Selectors
#     'sklearn.feature_selection.SelectFwe': {
#         'alpha': np.arange(0, 0.05, 0.001),
#         'score_func': {
#             'sklearn.feature_selection.f_regression': None
#         }
#     },

#     'sklearn.feature_selection.SelectPercentile': {
#         'percentile': range(1, 100),
#         'score_func': {
#             'sklearn.feature_selection.f_regression': None
#         }
#     },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.1, 0.25]
    },

    'sklearn.decomposition.PCA': {
        'n_components': [2, 3, 5, 10]
    },

    'sklearn.decomposition.FastICA': {
        'n_components': [2, 3, 5, 10]
    },

#     'sklearn.feature_selection.RFE': {
#         'step': np.arange(0.05, 1.01, 0.05),
#         'estimator': {
#             'sklearn.ensemble.ExtraTreesClassifier': {
#                 'n_estimators': [100],
#                 'criterion': ['gini', 'entropy'],
#                 'max_features': np.arange(0.05, 1.01, 0.05)
#             }
#         }
#     },

#     'sklearn.feature_selection.SelectFromModel': {
#         'threshold': np.arange(0, 1.01, 0.05),
#         'estimator': {
#             'sklearn.ensemble.ExtraTreesRegressor': {
#                 'n_estimators': [100],
#                 'max_features': np.arange(0.05, 1.01, 0.05)
#             }
#         }
#     }

}