import os
import sys
from dataclasses import dataclass 

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor  
)

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting Training and test input Data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet Regression": ElasticNet(),
                "Polynomial Regression": PolynomialFeatures(),
                "Support Vector Regression (SVR)": SVR(),
                "Decision Tree Regression": DecisionTreeRegressor(),
                "Random Forest Regression": RandomForestRegressor(),
                "Gradient Boosting Regression": GradientBoostingRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "K-Neighbors Regression": KNeighborsRegressor(),
                "Catboost Regression": CatBoostRegressor(verbose=False),
                "XGB Regression": XGBRegressor()
            }

            model_report:dict=evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            models=models)

            # Select best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # select best model from dist
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            if best_model_score<0.6:
                raise CustomException("Not any model fit for this data")
            raise logging.info("best found model on training and test dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(x_test)

            return r2_score

        except Exception as e:
            raise CustomException(e, sys)






