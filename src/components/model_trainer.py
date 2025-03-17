import os 
import sys 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import save_object,evaluate_models

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        logging.info("Model Training method started")
        try:
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error'],
                    'max_depth': [None],
                    'max_features': [None],
                    'min_samples_split': [2],
                    'random_state': [0]
                },
                "Random Forest":{
                    # 'criterion': ['squared_error'],
                    # 'max_depth': [None],
                    'max_features': ['sqrt'],
                    'min_samples_split': [2],
                    'n_estimators': [100, 500]
                },
                "Gradient Boosting":{
                    # 'loss': ['squared_error'],
                    'learning_rate': [0.1, 0.01],
                    'subsample': [1.0, 0.5],
                    'n_estimators': [50, 100]    
                },  
                "Linear Regression":{}, 
                "XGBRegressor":{
                    'learning_rate': [0.1, 0.01],   
                    'n_estimators': [8, 16]
                }
                ,
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                }
            }



            model_report:dict=evaluate_models(x_train,y_train,x_test,y_test,models, params)
            print(model_report)
            print("\n====================================================================================\n")
            logging.info(f"Model Report : {model_report}")

            # To get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset is: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            score =r2_score(y_test,predicted)
            return score 

        except Exception as e:    
            raise CustomException(e,sys)
            logging.error("Error in Model Training method")


