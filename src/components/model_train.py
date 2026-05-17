import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from src.exeception import CustomBreasctCancerException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModeltrainConfig:
    train_model_file_path=os.path.join("artifact",'model.pkl')
    
class ModelTrain:
    def __init__(self):
        self.model_train_config=ModeltrainConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models={
                'Logistic Regression':LogisticRegression(),
                'SVM':SVC(),
                'Random Forest':RandomForestClassifier(),
                'MLP Neural Network':MLPClassifier()
            }
            
            param={
                 "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"]
                },
                  "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                },
                   "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "criterion": ["gini", "entropy"]
                },

                "MLP Neural Network": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "activation": ["relu", "tanh"],
                    "learning_rate_init": [0.001, 0.01],
                }
            }
            
            model_report:dict=evaluate_models(
                X_train=X_train,y_train=y_train,
                X_test=X_test,y_test=y_test,
                models=models,param=param
            )
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score <0.7:
                raise CustomBreasctCancerException("No best model found")
            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")
            
            save_object(
                file_path=self.model_train_config.train_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            accscore = accuracy_score(y_test, predicted)
            return accscore
            
        except Exception as e:
            raise CustomBreasctCancerException(e,sys)      