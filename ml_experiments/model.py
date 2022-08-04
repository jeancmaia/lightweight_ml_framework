import pandas as pd
import numpy as np
import joblib 

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from xgboost import XGBClassifier

from scripts.features_handling import DEFAULT_FEATURES as original_features
from scripts.features_handling import run as raw_feature_preprocessing_handling


class BaseModelPersistence:
    def persist(self, output_dir: Path):
        joblib.dump(
            value=self,
            filename=output_dir+'/model.joblib'
        )
        
    def fit(self, X: pd.DataFrame, y: np.array):
        self.pipeline.fit(X, y)
        
    def predict_proba(self, X: pd.DataFrame):
        return self.pipeline.predict_proba(X)
   
    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)
    
class Model(BaseModelPersistence):
    def __init__(self):
        
        xgb_pipe = Pipeline() # To specify model
                
        self.pipeline = xgb_pipe
        
class ModelInterface(Model):
    def __init__(self):
        super().__init__()
        self.threshold = None
        
    def pipeline_prediction(self, X: pd.DataFrame):
        """Base method for prediction on raw data.

        Args:
            X (pd.DataFrame): Raw data for prediction
        """
        X.columns = original_features

        data, features = raw_feature_preprocessing_handling(X, original_features.copy())
        data = data[features]
        
        features_cleaned = [str(feature) for feature in features]

        data.columns = features_cleaned
        return self.pipeline.predict_proba(data)[0]
        