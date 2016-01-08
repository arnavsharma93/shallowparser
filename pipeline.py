from customutils import *
import numpy as np
from sklearn.base import BaseEstimator


class CMSTPipeline(BaseEstimator):
    '''
    Pipeline for parsing CMST
    It takes an argument steps - list of tuples where each tuple contains the name
    of the step and its instantiated model
    Last step has to predict
    '''
    def __init__(self, steps):
        self.final_step = steps.pop()
        self.steps = steps

    def fit(self, X, y):
        for name, model in self.steps:
            print '\tfitting and adding %s' % name
            model.fit(X, GetColumn(X, name))

        print '\tfitting %s' % self.final_step.name
        self.final_step.model.fit(X, y)

    def predict(self, X):
        _X = np.copy(X)
        for i, (name, model) in enumerate(self.steps):
            _X = OverwriteColumn(_X, model.predict(_X), name)

        y_pred = self.final_step.model.predict(_X)
        return y_pred




