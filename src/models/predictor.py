"""
Machine learning models for spread prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb


class SpreadPredictor:
    """Base class for spread prediction models."""
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize predictor.
        
        Args:
            model_type: 'gradient_boosting', 'xgboost', or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        print(f"✓ Model loaded from {path}")


class GradientBoostingPredictor(SpreadPredictor):
    """Gradient Boosting predictor."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_samples_split: int = 20
    ):
        """Initialize Gradient Boosting model."""
        super().__init__(model_type='gradient_boosting')
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """Train Gradient Boosting model."""
        print("Training Gradient Boosting model...")
        
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        print(f"Training R² score: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"Validation R² score: {val_score:.4f}")
            print(f"Validation RMSE: {val_rmse:.4f}")


class XGBoostPredictor(SpreadPredictor):
    """XGBoost predictor."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: int = 1
    ):
        """Initialize XGBoost model."""
        super().__init__(model_type='xgboost')
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            random_state=42
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """Train XGBoost model."""
        print("Training XGBoost model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"Training RMSE: {train_rmse:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"Validation RMSE: {val_rmse:.4f}")


class LightGBMPredictor(SpreadPredictor):
    """LightGBM predictor."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31
    ):
        """Initialize LightGBM model."""
        super().__init__(model_type='lightgbm')
        
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            verbose=-1
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """Train LightGBM model."""
        print("Training LightGBM model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if X_val is not None else None,
            eval_names=['train', 'val'] if X_val is not None else ['train']
        )
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"Training RMSE: {train_rmse:.4f}")
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"Validation RMSE: {val_rmse:.4f}")


class ModelTrainer:
    """Wrapper for training and evaluating models."""
    
    def __init__(self, model_type: str = 'gradient_boosting', **kwargs):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model to use
            **kwargs: Model-specific parameters
        """
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingPredictor(**kwargs)
        elif model_type == 'xgboost':
            self.model = XGBoostPredictor(**kwargs)
        elif model_type == 'lightgbm':
            self.model = LightGBMPredictor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_test_split_time_series(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data for time series (no shuffle).
        
        Args:
            X: Features
            y: Targets
            train_size: Fraction for training
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * train_size)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


if __name__ == "__main__":
    print("ML Models Module")
    print("-" * 50)
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randn(200)
    
    # Test each model
    for model_type in ['gradient_boosting', 'xgboost', 'lightgbm']:
        print(f"\nTesting {model_type}...")
        trainer = ModelTrainer(model_type=model_type)
        trainer.model.train(X_train, y_train, X_val, y_val)
        
        metrics = trainer.evaluate(X_val, y_val)
        print(f"Test metrics: {metrics}")
