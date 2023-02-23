from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch as th
from torch import nn
from .modeling import train, decision_tree_encoding, FeedForwardMLPResidual, NumericEmbedding

class BaseExperiment(ABC):
    def __init__(self,  Xs, ys):
        self.Xs = Xs
        self.ys = ys

    @abstractmethod
    def build_model(self, trial):
        pass 

    @abstractmethod
    def objective_(self):
        pass

    @abstractmethod
    def get_best_y_test(self, trial):
        pass


class RFExperiment(BaseExperiment):

    def build_model(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 30, 500)
        max_depth = trial.suggest_int('max_depth', 3, 12)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(self.Xs['train'], self.ys['train'])
        return clf
    
    def objective_(self):
        def objective(trial):
            clf = self.build_model(trial)
            return accuracy_score(self.ys['valid'], clf.predict(self.Xs['valid']))
        
        return objective

    def get_best_y_test(self, trial):
        clf = self.build_model(trial)
        return clf.predict_proba(self.Xs['test'])[:,1]


class XGBExperiment(RFExperiment):
    def build_model(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 30, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 12)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        
        bst = XGBClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate, 
            objective='binary:logistic',
            reg_alpha=2,
            early_stopping_rounds=40
        )
        bst.fit(
            self.Xs['train'], self.ys['train'], 
            eval_set=[
                (self.Xs['train'], self.ys['train']), 
                (self.Xs['valid'], self.ys['valid'])
            ]
        )
        return bst


class NNRawExperiment(BaseExperiment):
    
    def set_training_parms(self, N_epochs, stop_criteria):
        self.N_epochs = N_epochs
        self.stop_criteria = stop_criteria
        self.l1_lambda = 0.0

    def build_model(self, trial):
        hidden_dim = trial.suggest_int('hidden_dim', 4, 2048, log=True)
        dropout = trial.suggest_float('dropout', 0, 0.99)
        model = nn.Sequential(
            nn.BatchNorm1d(self.Xs['train'].shape[1]),
            FeedForwardMLPResidual(
                d_model=self.Xs['train'].shape[1], dim_feedforward=hidden_dim, dropout=dropout,
            ),
            nn.Linear(self.Xs['train'].shape[1], 1),
        )
        return model

    def objective_(self):
        def objective(trial):
            model = self.build_model(trial)
            _, _, _ = train(
                model, self.Xs['train'], self.ys['train'], 
                self.Xs['valid'], self.ys['valid'], self.Xs['test'], 
                self.N_epochs, stop_criteria=self.stop_criteria, 
                l1_lambda=self.l1_lambda,
            )
            return accuracy_score(self.ys['valid'], (model(self.Xs['valid'])>=0.5).int())
        return objective

    def get_best_y_test(self, trial):
        model = self.build_model(trial)
        train_losses, valid_losses, y_hat_test = train(
            model, self.Xs['train'], self.ys['train'], 
            self.Xs['valid'], self.ys['valid'], self.Xs['test'], 
            self.N_epochs, stop_criteria=self.stop_criteria
        )
        return train_losses, valid_losses, y_hat_test


class DTEExperiment(NNRawExperiment):
    def __init__(self,  Xs, ys):
        self.Xs_bk = Xs # originals used for growing trees
        self.Xs = None # will replace after growing trees
        self.ys = ys

    def build_model(self, trial):

        min_impurity_decrease = trial.suggest_float(
            'min_impurity_decrease', 1e-6, 1e-4, log=True
        ) # 1e-4

        Xs_dt, _, _ = decision_tree_encoding(
            self.Xs_bk, self.ys, min_impurity_decrease=min_impurity_decrease)

        self.Xs = Xs_dt

        n_uniques = (self.Xs['train'].max(dim=0)[0]+1).numpy()

        embed_dim = trial.suggest_int('embed_dim', 1, 32) # 8
        hidden_dim = trial.suggest_int('hidden_dim', 8, 2048, log=True) # 512
        dropout_embed = trial.suggest_float('dropout_embed', 0, 0.5) # 0.16 
        dropout_ffn = trial.suggest_float('dropout_ffn', 0, 0.5)
        self.l1_lambda = trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True) # 4e-4 
        model = nn.Sequential(
            NumericEmbedding(n_uniques, embed_dim, 'concat'),
            nn.Dropout(dropout_embed),
            FeedForwardMLPResidual(
                d_model=n_uniques.shape[0]*embed_dim, dim_feedforward=hidden_dim, dropout=dropout_ffn,
            ),
            nn.Linear(n_uniques.shape[0]*embed_dim, 1),
        )
        return model