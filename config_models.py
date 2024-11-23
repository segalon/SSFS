
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, \
    Ridge, SGDRegressor

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


XGBOOST_IMP_TYPES = ["gain", "total_gain", "weight", "cover"]

CLFS = [XGBClassifier, LogisticRegression, SVC, SGDClassifier, RandomForestClassifier]
REGS = [XGBRegressor, SVR, LinearRegression, RandomForestRegressor, SGDRegressor]

MODELS_DICT = {
    "xgboost_clf":
        {"class": XGBClassifier,
         "args": {"objective": "binary:logistic"}},
    "xgboost_reg":
        {"class": XGBRegressor,
         "args": {}},
    "xgboostl1_reg":
        {"class": XGBRegressor,
         "args": {"objective": "reg:absoluteerror"}},
    "linear_clf":
        {"class": LogisticRegression,
         "args": {"max_iter": 1000}},
    "linear_reg":
        {"class": LinearRegression,
         "args": {}},
    "ridge_reg":
        {"class": Ridge,
         "args": {}},
    "svc_clf":
        {"class": SVC,
         "args": {}},
    "linearl1_reg":
        {"class": SGDRegressor,
         "args": {"loss": "espilon_insensitive",
                  "epsilon": 0.0,
                  "max_iter": 1000}},
}



