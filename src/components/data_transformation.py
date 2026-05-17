from src.exeception import CustomBreasctCancerException
from src.logger import logging
from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifact",'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            num_col=['texture_mean', 'area_mean', 'smoothness_mean',
               'compactness_mean', 'concavity_mean', 'concave points_mean',
               'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'perimeter_se',
               'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
               'concave points_se', 'symmetry_se', 'fractal_dimension_se',
               'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
               'compactness_worst', 'concavity_worst', 'concave points_worst',
               'symmetry_worst', 'fractal_dimension_worst']
            
            num_pipe_line=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info("num_pipeline are done")
            
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipe_line,num_col)
                ]
            )
            
            return preprocessor
            
            
        except Exception as e:
            raise CustomBreasctCancerException(e,sys)        
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "diagnosis"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            if train_df[target_column_name].dtype == 'object' or isinstance(train_df[target_column_name].iloc[0], str):
                target_feature_train_df = train_df[target_column_name].map({'M': 1, 'B': 0})
            else:
                target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            if test_df[target_column_name].dtype == 'object' or isinstance(test_df[target_column_name].iloc[0], str):
                target_feature_test_df = test_df[target_column_name].map({'M': 1, 'B': 0})
            else:
                target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomBreasctCancerException(e, sys)    