import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os
from typing import Dict, List, Any

class ProviderPerformanceForecaster:
    """
    Machine Learning model to forecast provider performance and reliability
    """
    
    def __init__(self, model_path='provider_performance_model.joblib'):
        """
        Initialize Provider Performance Forecaster
        
        Args:
            model_path (str): Path to save/load trained model
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path
        self.performance_model = None
        self.reliability_model = None
        self.scaler = StandardScaler()
        
        # Feature definitions
        self.features = [
            'total_tasks_completed',
            'average_task_duration',
            'task_success_rate',
            'computational_power',
            'historical_reputation',
            'network_contribution'
        ]
        
        self.logger.info("Provider Performance Forecaster initialized")

    def _preprocess_data(self, providers_data: pd.DataFrame):
        """
        Preprocess provider performance data for model training
        
        Args:
            providers_data (pd.DataFrame): Historical provider performance data
        
        Returns:
            Preprocessed features and targets
        """
        # Select features
        X = providers_data[self.features]
        
        # Separate performance and reliability targets
        y_performance = providers_data['future_performance_score']
        y_reliability = providers_data['future_reliability_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_performance, y_reliability

    def train_models(self, providers_data: pd.DataFrame):
        """
        Train performance and reliability prediction models
        
        Args:
            providers_data (pd.DataFrame): Historical provider performance data
        """
        try:
            # Preprocess data
            X, y_performance, y_reliability = self._preprocess_data(providers_data)
            
            # Split data
            X_train, X_test, y_perf_train, y_perf_test, \
            y_rel_train, y_rel_test = train_test_split(
                X, y_performance, y_reliability, test_size=0.2, random_state=42
            )
            
            # Train performance prediction model
            self.performance_model = GradientBoostingRegressor(n_estimators=100)
            self.performance_model.fit(X_train, y_perf_train)
            
            # Evaluate performance model
            perf_predictions = self.performance_model.predict(X_test)
            perf_mae = mean_absolute_error(y_perf_test, perf_predictions)
            perf_r2 = r2_score(y_perf_test, perf_predictions)
            
            self.logger.info(f"Performance Prediction MAE: {perf_mae:.4f}")
            self.logger.info(f"Performance Prediction R2 Score: {perf_r2:.4f}")
            
            # Train reliability prediction model
            self.reliability_model = GradientBoostingRegressor(n_estimators=100)
            self.reliability_model.fit(X_train, y_rel_train)
            
            # Evaluate reliability model
            rel_predictions = self.reliability_model.predict(X_test)
            rel_mae = mean_absolute_error(y_rel_test, rel_predictions)
            rel_r2 = r2_score(y_rel_test, rel_predictions)
            
            self.logger.info(f"Reliability Prediction MAE: {rel_mae:.4f}")
            self.logger.info(f"Reliability Prediction R2 Score: {rel_r2:.4f}")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise

    def predict_provider_performance(self, provider_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict future performance and reliability of a provider
        
        Args:
            provider_features (dict): Features of the provider
        
        Returns:
            dict: Predicted performance and reliability scores
        """
        try:
            # Prepare input features
            input_data = pd.DataFrame([provider_features])
            
            # Scale features
            input_scaled = self.scaler.transform(input_data[self.features])
            
            # Predict performance
            performance_prediction = self.performance_model.predict(input_scaled)
            reliability_prediction = self.reliability_model.predict(input_scaled)
            
            return {
                'predicted_performance_score': performance_prediction[0],
                'predicted_reliability_score': reliability_prediction[0]
            }
        
        except Exception as e:
            self.logger.error(f"Provider performance prediction error: {e}")
            raise

    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save models
            joblib.dump({
                'performance_model': self.performance_model,
                'reliability_model': self.reliability_model,
                'scaler': self.scaler
            }, self.model_path)
            
            self.logger.info(f"Models saved to {self.model_path}")
        
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
            raise

    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            if os.path.exists(self.model_path):
                saved_models = joblib.load(self.model_path)
                
                self.performance_model = saved_models['performance_model']
                self.reliability_model = saved_models['reliability_model']
                self.scaler = saved_models['scaler']
                
                self.logger.info("Pre-trained models loaded successfully")
                return True
            
            self.logger.warning("No pre-trained models found")
            return False
        
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            raise

def generate_synthetic_provider_data(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic provider performance data for model training
    
    Args:
        num_samples (int): Number of synthetic provider samples
    
    Returns:
        pd.DataFrame: Synthetic provider performance dataset
    """
    np.random.seed(42)
    
    data = {
        'total_tasks_completed': np.random.randint(10, 1000, num_samples),
        'average_task_duration': np.random.uniform(0.1, 24, num_samples),
        'task_success_rate': np.random.uniform(0.5, 1, num_samples),
        'computational_power': np.random.uniform(1, 100, num_samples),
        'historical_reputation': np.random.uniform(0.5, 1, num_samples),
        'network_contribution': np.random.uniform(0.1, 1, num_samples),
        'future_performance_score': np.random.uniform(0.5, 1, num_samples),
        'future_reliability_score': np.random.uniform(0.5, 1, num_samples)
    }
    
    return pd.DataFrame(data)

def main():
    # Generate synthetic training data
    synthetic_data = generate_synthetic_provider_data()
    
    # Initialize forecaster
    forecaster = ProviderPerformanceForecaster()
    
    # Train models
    forecaster.train_models(synthetic_data)
    
    # Example provider for prediction
    sample_provider = {
        'total_tasks_completed': 500,
        'average_task_duration': 2.5,
        'task_success_rate': 0.85,
        'computational_power': 75,
        'historical_reputation': 0.9,
        'network_contribution': 0.7
    }
    
    # Predict provider performance
    performance_result = forecaster.predict_provider_performance(sample_provider)
    print("Provider Performance Prediction:", performance_result)

if __name__ == '__main__':
    main()
