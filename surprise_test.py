# -*- coding:utf-8 -*-
"""
-------------------------------------------------
   File Name:     surprise_test
   Description:   个推系统使用demo
   Author:        Miller
   date：         2017/9/11 0011
-------------------------------------------------
"""
__author__ = 'Miller'

from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)
