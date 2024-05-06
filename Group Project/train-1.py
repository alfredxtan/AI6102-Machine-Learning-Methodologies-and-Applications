import warnings
from collections import defaultdict

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import GenericUnivariateSelect as GUS, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



warnings.filterwarnings("ignore")

train = pd.read_csv("data/train.csv", index_col="Id")
train.columns = train.columns.str.strip()
train["EJ"] = train["EJ"].map({"A": 0.0, "B": 1.0})

NULL_COLS = train.columns[train.isnull().any()].to_list()
TARGET_COL = "Class"
FEATURE_COLS = train.columns.drop(TARGET_COL).to_list()
CAT_FEATURE_COLS = ["EJ"]
NUM_FEATURE_COLS = [col for col in FEATURE_COLS if col not in CAT_FEATURE_COLS]


def knn_impute(data, *args, **kwargs):
    imp = KNNImputer(*args, **kwargs)
    return pd.DataFrame(imp.fit_transform(data), columns=data.columns, index=data.index)


train = knn_impute(train)


def generate_engineered_features(data, feature_cols):
    for i, col_1 in enumerate(feature_cols, 1):
        for col_2 in feature_cols[i:]:
            data[f"{col_1}_{col_2}_DIFF"] = data[col_1] - data[col_2]
            data[f"{col_1}_{col_2}_SUM"] = data[col_1] + data[col_2]
            data[f"{col_1}_{col_2}_PRODUCT"] = data[col_1] * data[col_2]
            data[f"{col_1}_{col_2}_RATIO"] = data[col_1] / (data[col_2] + 1)
    return data


RATIO_FEATURE_COLS = [
    f"{col_1}_{col_2}_RATIO"
    for i, col_1 in enumerate(NUM_FEATURE_COLS, 1)
    for col_2 in NUM_FEATURE_COLS[i:]
]
PRODUCT_FEATURE_COLS = [
    f"{col_1}_{col_2}_PRODUCT"
    for i, col_1 in enumerate(NUM_FEATURE_COLS, 1)
    for col_2 in NUM_FEATURE_COLS[i:]
]
SUM_FEATURE_COLS = [
    f"{col_1}_{col_2}_SUM"
    for i, col_1 in enumerate(NUM_FEATURE_COLS, 1)
    for col_2 in NUM_FEATURE_COLS[i:]
]
DIFF_FEATURE_COLS = [
    f"{col_1}_{col_2}_DIFF"
    for i, col_1 in enumerate(NUM_FEATURE_COLS, 1)
    for col_2 in NUM_FEATURE_COLS[i:]
]

train = generate_engineered_features(train, NUM_FEATURE_COLS)

df = train.copy()
X = df[
    FEATURE_COLS
    + RATIO_FEATURE_COLS
    + PRODUCT_FEATURE_COLS
    + SUM_FEATURE_COLS
    + DIFF_FEATURE_COLS
].values
y = df[TARGET_COL].values
print(X.shape)


#sel2 = GUS(mutual_info_classif, mode = 'k_best', param = 50)
sel1 = GUS(f_classif, mode = 'k_best', param = 50)
X = sel2.fit_transform(X, y)
print(X.shape)

#Standardize before PCA
scaler = StandardScaler()
X = scaler.fit_transform(X)
sel1 = PCA(n_components = 50)  #max n_components = 617
X = sel1.fit_transform(X)
print(X.shape)



SELECTED_FEATURES_1 = sel1.get_feature_names_out(
    FEATURE_COLS
    + RATIO_FEATURE_COLS
    + PRODUCT_FEATURE_COLS
    + SUM_FEATURE_COLS
    + DIFF_FEATURE_COLS).tolist()




def balanced_log_loss(y_true, y_pred, **kwargs):
    y_pred_clipped = np.clip(y_pred.reshape(-1), 1e-15, 1 - 1e-15)
    y_true_ = y_true.reshape(-1).astype(np.float64)
    class_counts = np.bincount(y_true.astype(np.int64))
    balanced_log_loss_score = (
        -((1 - y_true_) * np.log(1 - y_pred_clipped)).sum() / class_counts[0]
        - (y_true_ * np.log(y_pred_clipped)).sum() / class_counts[1]
    ) / 2
    return balanced_log_loss_score


balanced_log_loss_metric = make_scorer(
    balanced_log_loss,
    response_method="predict_proba",
    greater_is_better=False,
)


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),  # 4508
        "depth": trial.suggest_int("depth", 3, 10),  # 4511
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),  # 4514
        "rsm": trial.suggest_float("rsm", 0.01, 1.0, log=True),  # 4528
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 0, 8),  # 4654
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),  # 4741
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),  # 4756
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),  # 4811
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 10
        )
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    model = CatBoostClassifier(
        **params, verbose=False, auto_class_weights="Balanced", random_seed=42
    )
    return cross_val_score(
        model, X, y, scoring=balanced_log_loss_metric, cv=5, n_jobs=1
    ).mean()


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=5000, n_jobs=1, show_progress_bar=True)
#study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)


print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


best_model = CatBoostClassifier(
    **study.best_params, verbose=False, auto_class_weights="Balanced", random_seed=42
)
print(study.best_params)
print(
    cross_val_score(
        best_model, X, y, scoring=balanced_log_loss_metric, cv=5, n_jobs=1
    ).mean()
)

rfecv = RFECV(
    estimator=best_model,
    step=1,
    min_features_to_select=1,
    cv=5,
    scoring=balanced_log_loss_metric,
    verbose=1,
    n_jobs=1,
)
rfecv.fit(X, y)

X = X[:, rfecv.support_]
print(X.shape)

SELECTED_FEATURES_3 = [
    col for col, support in zip(SELECTED_FEATURES_1, rfecv.support_) if support
]
print(SELECTED_FEATURES_3)

print(
    cross_val_score(
        best_model, X, y, scoring=balanced_log_loss_metric, cv=5, n_jobs=1
    ).mean()
)
