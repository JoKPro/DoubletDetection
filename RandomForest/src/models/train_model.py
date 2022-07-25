import tensorflow_decision_forests as tfdf
from tensorflow import metrics
import os
import sys
import numpy as np
import datatable as dt
from datatable import f
import pandas as pd
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix



data = pd.read_csv("../../data/processed/pbmc_hvg_12012_100.csv")


def split_dataset(dataset, test_ratio=0.3):
    """splits a panda dataframe in two"""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


class RandomForestModel:
    def __init__(self, data, obs_col, class_col):
        self.data = data
        self.obs_col = obs_col
        self.class_col = class_col
        self.train_ds = None
        self.test_ds = None
        self.models = dict()

    def train_test_split(self, ratio=0.3):
        if self.data:
            test_indices = np.random.rand(len(self.data)) < ratio
            train_set, test_set = self.data[~test_indices], self.data[test_indices]
            self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_set, label=self.class_col)
            self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_set, label=self.class_col)
        else:
            print("data not loaded")

    def add_new_model(self, model_name, params=None):

        features = self.train_df.drop(["obs", "cell_type"], axis=1).columns
        features = tfdf.keras.FeatureUsage(features)

        if params:
            print("parameter supplied")
            model = tfdf.keras.RandomForestModel(
                features=features,
                exclude_non_specified_features=True,
                random_seed=params["random_seed"],
                num_trees=params["num_trees"],
                categorical_algorithm=params["categorical_algorithm"],
                compute_oob_performances=params["compute_oob_performances"],
                growing_strategy=params["growing_strategy"],
                honest=params["honest"],
                max_depth=params["max_depth"],
                max_num_nodes=params["max_num_nodes"]
            )
        else:
            print("no parameters supplied, using default model")
            model = tfdf.keras.RandomForestModel()
        self.models.update({model_name: model})
        print(f"{model_name} added")

    def train_model(self, model_name, metrics="Accuracy", verbose=False):
        if model_name in self.models.keys():
            model = self.models[model_name]
            model.fit(x=self.train_ds,  verbose=verbose)

            model.compile(metrics=metrics)
        else:
            print("model not found.")

    def evaluate_model(self, model_name):
        model = self.models[model_name]
        model.evaluate(test_ds)
        print()
        for name, value in evaluation.items():
            print(f"{name}: {value:.4f}")

    # def predict(self, model_name):
    #     model = self.models[model_name]
    #     prediction = model.predict(self.test_ds)
    #     print('Classification Report: \n')
    #     print(classification_report(y_test, rf_prediction))
    #     print('\nConfusion Matrix: \n')
    #     print(confusion_matrix(y_test, rf_prediction))




params_v1 = {
    "random_seed": 123456,
    "num_trees": 500,
    "categorical_algorithm": "CART",
    "compute_oob_performances": False,
    "growing_strategy": "LOCAL",
    "honest": False,
    "max_depth": 16,
    "max_num_nodes": None,
}



rf_models = RandomForestModel(data, obs_col="obs", class_coll="cell_type")



for







# training_set, test_set = split_dataset(data)
# train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(training_set, label="cell_type")
# test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_set, label="cell_type")
#
# model_1 = tfdf.keras.RandomForestModel()
# model_1.fit(x=train_ds, verbose=2)
#
# model_1.compile(metrics=["Accuracy"])
# evaluation = model_1.evaluate(test_ds, return_dict=True)
# print()
# for name, value in evaluation.items():
#     print(f"{name}: {value:.4f}")
#
# model_1.summary()
# model_1.make_inspector().evaluation()
#
# logs = model_1.make_inspector().training_logs()
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
# plt.xlabel("Number of trees")
# plt.ylabel("Accuracy (out-of-bag)")
#
# plt.subplot(1, 2, 2)
# plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
# plt.xlabel("Number of trees")
# plt.ylabel("Logloss (out-of-bag)")
#
# plt.show()
