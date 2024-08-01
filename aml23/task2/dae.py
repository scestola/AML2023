import os
import optuna
import sklearn
import pandas as pd 
import numpy as np 
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


base_path="data"
filename="dae_384"

feature_dir = os.path.join(base_path, filename + ".npy")
target_dir = os.path.join(base_path, filename + "_target.npy")
features = np.load(open(feature_dir, "rb"))
targets = np.load(open(target_dir, "rb")).ravel()
num_train = targets.shape[0]
train_features = features[:num_train, :]
test_features = features[num_train:, :]


def cross_validation(model):
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(train_features, targets):
        y_true = targets[val_idx]
        y_pred = sklearn.base.clone(model).fit(
            train_features[train_idx], targets[train_idx]
        ).predict(train_features[val_idx])
        score = f1_score(y_true, y_pred, average="micro")
        print(f"{score:.5f}")
        scores.append(score)

    cv = np.mean(scores)
    return cv
    

def main():
    model = BaggingClassifier(
        base_estimator=RidgeClassifier(alpha=1.0),
        n_estimators=10
    )
    
    cv = cross_validation(model)
    print(f"CV: {cv:.5f}")


if __name__ == "__main__":
    main()