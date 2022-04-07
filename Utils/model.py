import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class PipelinePredictor(BaseEstimator, TransformerMixin):

    def __init__(self, path):
        self.model = joblib.load(path)
        self.best_estimator_ = self.model.best_estimator_

    def predict(self, x):
        # do pre-processing stuff
        pred = self.model.predict(x)
        proba = self.model.predict_proba(x)
        # do post-processing stuff
        return pred, proba