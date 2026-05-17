import sys
import os 
import pickle
import pandas as pd
import numpy as np
from src.exeception import CustomBreasctCancerException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import dill

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomBreasctCancerException(e,sys)   
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            rd=RandomizedSearchCV(model,para,cv=3)
            rd.fit(X_train,y_train)
            model.set_params(**rd.best_params_)
            model.fit(X_train, y_train)
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            test_acc_score=accuracy_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_acc_score
            
        return report    
            
    except Exception as e:
        raise CustomBreasctCancerException(e,sys)     


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            try:
                return dill.load(file_obj)
            except Exception:
                file_obj.seek(0)
                return pickle.load(file_obj)
    except Exception as e:
        raise CustomBreasctCancerException(e, sys)    
