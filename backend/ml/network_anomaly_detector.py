import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import logging
import joblib
import os
from typing import List, Dict, Any, Tuple

class NetworkAnomalyDetector:
    """
    Advanced anomaly detection system for decentralized AI computation network
    """
    
    def __init__(self, contamination: float = 0.1, model_path: str = 'anomaly_detection_model.joblib'):
        """
        Initialize Network Anomaly Detector
        
        Args:
            contamination (float): Expected proportion of anomalies in the dataset
            model_path (str): Path to save/load trained models
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Anomaly detection parameters
        self.contamination = contamination
        self.model_path = model_path
        
        # Machine learning models
        self.isolation_forest = None
        self.pca_transformer = None
        self.scaler = StandardScaler()
        
        # Feature definitions
        self.features = [
            'computational_power',
            'task_completion_rate',
            'average_task_duration',
            'network_latency',
            'resource_utilization',
            'error_rate'
        ]
        
        self.logger.info("Network Anomaly Detector initialized")

    def _preprocess_data(self, network_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess network performance data
        
        Args:
            network_data (pd.DataFrame): Network performance metrics
        
        Returns:
            np.ndarray: Preprocessed and scaled feature matrix
        """
        # Select and scale features
        X = network_data[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        self.pca_transformer = PCA(n_components=0.95)  # Preserve 95% variance
        X_reduced = self.pca_transformer.fit_transform(X_scaled)
        
        return X_reduced

    def train_anomaly_detection_model(self, network_data: pd.DataFrame):
        """
        Train anomaly detection model using Isolation Forest
        
        Args:
            network_data (pd.DataFrame): Historical network performance data
        """
        try:
            # Preprocess data
            X_processed = self._preprocess_data(network_data)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination, 
                random_state=42
            )
            self.isolation_forest.fit(X_processed)
            
            # Save models
            self._save_models()
            
            self.logger.info("Anomaly detection model trained successfully")
        
        except Exception as e:
            self.logger.error(f"Anomaly detection model training error: {e}")
            raise

    def detect_network_anomalies(self, current_network_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in current network performance
        
        Args:
            current_network_data (pd.DataFrame): Current network performance metrics
        
        Returns:
            Dict: Anomaly detection results
        """
        try:
            # Preprocess current data using trained scaler and PCA
            X_current = current_network_data[self.features]
            X_scaled = self.scaler.transform(X_current)
            X_reduced = self.pca_transformer.transform(X_scaled)
            
            # Predict anomalies
            anomaly_labels = self.isolation_forest.predict(X_reduced)
            anomaly_scores = -self.isolation_forest.score_samples(X_reduced)
            
            # Identify anomalous data points
            anomalies = current_network_data[anomaly_labels == -1]
            
            results = {
                'total_data_points': len(current_network_data),
                'anomalies_detected': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(current_network_data) * 100,
                'anomalous_providers': anomalies.to_dict('records'),
                'anomaly_scores': dict(zip(anomalies.index, anomaly_scores[anomaly_labels == -1]))
            }
            
            self.logger.info(f"Detected {results['anomalies_detected']} anomalies")
            return results
        
        except Exception as e:
            self.logger.error(f"Network anomaly detection error: {e}")
            raise

    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save models
            joblib.dump({
                'isolation_forest': self.isolation_forest,
                'pca_transformer': self.pca_transformer,
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
                
                self.isolation_forest = saved_models['isolation_forest']
                self.pca_transformer = saved_models['pca_transformer']
                self.scaler = saved_models['scaler']
                
                self.logger.info("Pre-trained models loaded successfully")
                return True
            
            self.logger.warning("No pre-trained models found")
            return False
        
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            raise

    def generate_anomaly_report(self, anomaly_results: Dict[str, Any]) -> str:
        """
        Generate a detailed anomaly detection report
        
        Args:
            anomaly_results (Dict): Anomaly detection results
        
        Returns:
            str: Formatted anomaly report
        """
        report = "Network Anomaly Detection Report\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Total Data Points: {anomaly_results['total_data_points']}\n"
        report += f"Anomalies Detected: {anomaly_results['anomalies_detected']}\n"
        report += f"Anomaly Percentage: {anomaly_results['anomaly_percentage']:.2f}%\n\n"
        
        report += "Anomalous Providers:\n"
        for provider in anomaly_results['anomalous_providers']:
            report += f"Provider ID: {provider.get('provider_id', 'Unknown')}\n"
            report += f"  Anomaly Score: {anomaly_results['anomaly_scores'].get(provider.get('index', -1), 'N/A')}\n"
            report += "  Suspicious Metrics:\n"
            for feature in self.features:
                report += f"    {feature}: {provider.get(feature, 'N/A')}\n"
            report += "\n"
        
        return report

def generate_synthetic_network_data(num_samples: int = 1000, anomaly_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic network performance data with anomalies
    
    Args:
        num_samples (int): Total number of data points
        anomaly_ratio (float): Proportion of anomalous data points
    
    Returns:
        Tuple of training and current network data
    """
    np.random.seed(42)
    
    # Normal data generation
    normal_samples = int(num_samples * (1 - anomaly_ratio))
    normal_data = {
        'computational_power': np.random.normal(50, 10, normal_samples),
        'task_completion_rate': np.random.normal(0.9, 0.05, normal_samples),
        'average_task_duration': np.random.normal(10, 2, normal_samples),
        'network_latency': np.random.normal(50, 10, normal_samples),
        'resource_utilization': np.random.normal(0.7, 0.1, normal_samples),
        'error_rate': np.random.normal(0.05, 0.02, normal_samples)
    }
    
    # Anomaly data generation
    anomaly_samples = num_samples - normal_samples
    anomaly_data = {
        'computational_power': np.random.normal(20, 20, anomaly_samples),
        'task_completion_rate': np.random.normal(0.5, 0.2, anomaly_samples),
        'average_task_duration': np.random.normal(30, 10, anomaly_samples),
        'network_latency': np.random.normal(200, 50, anomaly_samples),
        'resource_utilization': np.random.normal(0.2, 0.2, anomaly_samples),
        'error_rate': np.random.normal(0.3, 0.1, anomaly_samples)
    }
    
    # Combine data
    training_data = pd.DataFrame(normal_data)
    current_data = pd.DataFrame({
        **normal_data,
        **anomaly_data
    })
    
    return training_data, current_data

def main():
    # Generate synthetic network data
    training_data, current_data = generate_synthetic_network_data()
    
    # Initialize anomaly detector
    anomaly_detector = NetworkAnomalyDetector(contamination=0.1)
    
    # Train anomaly detection model
    anomaly_detector.train_anomaly_detection_model(training_data)
    
    # Detect network anomalies
    anomaly_results = anomaly_detector.detect_network_anomalies(current_data)
    
    # Generate and print anomaly report
    anomaly_report = anomaly_detector.generate_anomaly_report(anomaly_results)
    print(anomaly_report)

if __name__ == '__main__':
    main()
