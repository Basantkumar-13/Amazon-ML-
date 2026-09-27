import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path


class MatcherModel:
    def __init__(self, random_seed=42):
        self.model = None
        self.seed = random_seed
        self.threshold = 0.5  # will be tuned later
    
    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=self.seed, stratify=y
        )
        
        # Count class balance for scale_pos_weight
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / max(n_pos, 1)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 30,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'scale_pos_weight': spw,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': self.seed,
            'n_jobs': -1,
        }
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        print(f"Training done. Best iteration: {self.model.best_iteration}")
        print(f"  Scale pos weight: {spw:.2f} (pos={n_pos:,}, neg={n_neg:,})")
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_match(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold
        probs = self.predict(X)
        return probs >= threshold
    
    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'threshold': self.threshold}, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        data = joblib.load(filepath)
        self.model = data['model']
        self.threshold = data['threshold']
        print(f"Model loaded from {filepath}")
        return self
