import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import logging
import json
import os

class TaskComplexityPredictor:
    """
    Machine Learning model to predict task complexity and resource requirements
    """
    
    def __init__(self, model_path='task_complexity_model.joblib'):
        """
        Initialize Task Complexity Predictor
        
        Args:
            model_path (str): Path to save/load trained model
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path
        self.complexity_model = None
        self.resource_model = None
        self.scaler = StandardScaler()
        
        # Feature definitions
        self.features = [
            'input_size',
            'computational_complexity',
            'data_type',
            'model_type',
            'previous_task_duration',
            'provider_historical_performance'
        ]
        
        self.logger.info("Task Complexity Predictor initialized")

    def _preprocess_data(self, tasks_data):
        """
        Preprocess task data for model training
        
        Args:
            tasks_data (pd.DataFrame): Historical task data
        
        Returns:
            Preprocessed features and targets
        """
        # Encode categorical features
        tasks_data['data_type'] = pd.Categorical(tasks_data['data_type']).codes
        tasks_data['model_type'] = pd.Categorical(tasks_data['model_type']).codes
        
        # Select features
        X = tasks_data[self.features]
        
        # Separate complexity and resource targets
        y_complexity = tasks_data['complexity_level']
        y_resources = tasks_data['required_compute_resources']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_complexity, y_resources

    def train_models(self, tasks_data):
        """
        Train complexity and resource prediction models
        
        Args:
            tasks_data (pd.DataFrame): Historical task data
        """
        try:
            # Preprocess data
            X, y_complexity, y_resources = self._preprocess_data(tasks_data)
            
            # Split data
            X_train, X_test, y_complexity_train, y_complexity_test, \
            y_resources_train, y_resources_test = train_test_split(
                X, y_complexity, y_resources, test_size=0.2, random_state=42
            )
            
            # Train complexity classification model
            self.complexity_model = RandomForestClassifier(n_estimators=100)
            self.complexity_model.fit(X_train, y_complexity_train)
            
            # Evaluate complexity model
            complexity_predictions = self.complexity_model.predict(X_test)
            complexity_accuracy = accuracy_score(y_complexity_test, complexity_predictions)
            self.logger.info(f"Complexity Prediction Accuracy: {complexity_accuracy:.2%}")
            self.logger.info(classification_report(y_complexity_test, complexity_predictions))
            
            # Train resource regression model
            self.resource_model = RandomForestRegressor(n_estimators=100)
            self.resource_model.fit(X_train, y_resources_train)
            
            # Evaluate resource model
            resource_predictions = self.resource_model.predict(X_test)
            resource_mae = mean_absolute_error(y_resources_test, resource_predictions)
            self.logger.info(f"Resource Prediction MAE: {resource_mae:.2f}")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise

    def predict_task_complexity(self, task_features):
        """
        Predict task complexity
        
        Args:
            task_features (dict): Features of the task
        
        Returns:
            str: Predicted complexity level
        """
        try:
            # Prepare input features
            input_data = pd.DataFrame([task_features])
            input_data['data_type'] = pd.Categorical(input_data['data_type']).codes
            input_data['model_type'] = pd.Categorical(input_data['model_type']).codes
            
            # Scale features
            input_scaled = self.scaler.transform(input_data[self.features])
            
            # Predict complexity
            complexity_prediction = self.complexity_model.predict(input_scaled)
            complexity_proba = self.complexity_model.predict_proba(input_scaled)
            
            return {
                'complexity_level': complexity_prediction[0],
                'complexity_probabilities': complexity_proba[0]
            }
        
        except Exception as e:
            self.logger.error(f"Complexity prediction error: {e}")
            raise

    def predict_resource_requirements(self, task_features):
        """
        Predict computational resource requirements
        
        Args:
            task_features (dict): Features of the task
        
        Returns:
            dict: Predicted resource requirements
        """
        try:
            # Prepare input features
            input_data = pd.DataFrame([task_features])
            input_data['data_type'] = pd.Categorical(input_data['data_type']).codes
            input_data['model_type'] = pd.Categorical(input_data['model_type']).codes
            
            # Scale features
            input_scaled = self.scaler.transform(input_data[self.features])
            
            # Predict resources
            resource_prediction = self.resource_model.predict(input_scaled)
            
            return {
                'required_compute_resources': resource_prediction[0],
                'estimated_duration': resource_prediction[0] * 0.5  # Example mapping
            }
        
        except Exception as e:
            self.logger.error(f"Resource prediction error: {e}")
            raise

    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save models
            joblib.dump({
                'complexity_model': self.complexity_model,
                'resource_model': self.resource_model,
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
                
                self.complexity_model = saved_models['complexity_model']
                self.resource_model = saved_models['resource_model']
                self.scaler = saved_models['scaler']
                
                self.logger.info("Pre-trained models loaded successfully")
                return True
            
            self.logger.warning("No pre-trained models found")
            return False
        
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            raise

def generate_synthetic_task_data(num_samples=1000):
    """
    Generate synthetic task data for model training
    
    Args:
        num_samples (int): Number of synthetic task samples
    
    Returns:
        pd.DataFrame: Synthetic task dataset
    """
    np.random.seed(42)
    
    data = {
        'input_size': np.random.uniform(1, 1000, num_samples),
        'computational_complexity': np.random.uniform(0.1, 10, num_samples),
        'data_type': np.random.choice(['image', 'text', 'numeric', 'time_series'], num_samples),
        'model_type': np.random.choice(['classification', 'regression', 'clustering', 'generation'], num_samples),
        'previous_task_duration': np.random.uniform(0.1, 24, num_samples),
        'provider_historical_performance': np.random.uniform(0.5, 1, num_samples),
        'complexity_level': np.random.choice(['low', 'medium', 'high'], num_samples),
        'required_compute_resources': np.random.uniform(1, 100, num_samples)
    }
    
    return pd.DataFrame(data)

def main():
    # Generate synthetic training data
    synthetic_data = generate_synthetic_task_data()
    
    # Initialize predictor
    predictor = TaskComplexityPredictor()
    
    # Train models
    predictor.train_models(synthetic_data)
    
    # Example task for prediction
    sample_task = {
        'input_size': 500,
        'computational_complexity': 5.5,
        'data_type': 'image',
        'model_type': 'classification',
        'previous_task_duration': 2.5,
        'provider_historical_performance': 0.85
    }
    
    # Predict complexity
    complexity_result = predictor.predict_task_complexity(sample_task)
    print("Task Complexity Prediction:", complexity_result)
    
    # Predict resource requirements
    resource_result = predictor.predict_resource_requirements(sample_task)
    print("Resource Requirements Prediction:", resource_result)

if __name__ == '__main__':
    main()
