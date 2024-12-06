import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class ProviderMatcher:
    def __init__(self):
        self.providers_database = pd.DataFrame(columns=[
            'provider_id', 
            'cpu_cores', 
            'memory_gb', 
            'gpu_enabled', 
            'gpu_memory', 
            'network_bandwidth', 
            'reliability_score', 
            'price_per_hour'
        ])
        self.scaler = StandardScaler()

    def register_provider(self, provider_details):
        """
        Register a new compute provider with detailed specifications.
        
        Args:
            provider_details (dict): Comprehensive provider specifications
        """
        new_provider = pd.DataFrame([provider_details])
        self.providers_database = pd.concat([self.providers_database, new_provider], ignore_index=True)

    def find_best_providers(self, task_requirements, top_n=3):
        """
        Find the most suitable providers for a given task.
        
        Args:
            task_requirements (dict): Computational task requirements
            top_n (int): Number of top providers to return
        
        Returns:
            list: Top N provider IDs most suitable for the task
        """
        # Preprocess providers data
        features = self.providers_database.drop('provider_id', axis=1)
        scaled_features = self.scaler.fit_transform(features)
        
        # Task requirements vector
        task_vector = np.array([
            task_requirements.get('cpu_cores', 1),
            task_requirements.get('memory_gb', 4),
            1 if task_requirements.get('gpu_required', False) else 0,
            task_requirements.get('gpu_memory', 0),
            task_requirements.get('network_bandwidth', 100)
        ]).reshape(1, -1)
        
        # Compute similarity scores
        similarity_scores = cosine_similarity(task_vector, scaled_features)[0]
        
        # Rank providers
        provider_rankings = pd.DataFrame({
            'provider_id': self.providers_database['provider_id'],
            'similarity_score': similarity_scores,
            'reliability_score': self.providers_database['reliability_score'],
            'price_score': 1 / self.providers_database['price_per_hour']
        })
        
        # Weighted ranking
        provider_rankings['composite_score'] = (
            0.5 * provider_rankings['similarity_score'] + 
            0.3 * provider_rankings['reliability_score'] + 
            0.2 * provider_rankings['price_score']
        )
        
        top_providers = provider_rankings.nlargest(top_n, 'composite_score')
        return top_providers['provider_id'].tolist()

    def cluster_providers(self, n_clusters=5):
        """
        Cluster providers based on their computational characteristics.
        
        Args:
            n_clusters (int): Number of provider clusters
        
        Returns:
            dict: Cluster assignments for each provider
        """
        features = self.providers_database.drop('provider_id', axis=1)
        scaled_features = self.scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        return {
            'provider_clusters': dict(zip(
                self.providers_database['provider_id'], 
                cluster_labels
            )),
            'cluster_centroids': kmeans.cluster_centers_
        }

    def predict_provider_performance(self, provider_id):
        """
        Predict future performance of a provider.
        
        Args:
            provider_id (str): Unique identifier of the provider
        
        Returns:
            dict: Performance prediction metrics
        """
        provider_data = self.providers_database[
            self.providers_database['provider_id'] == provider_id
        ]
        
        if provider_data.empty:
            return None
        
        # Simple performance prediction based on current specs
        performance_prediction = {
            'reliability_forecast': provider_data['reliability_score'].values[0] * 1.1,
            'capacity_utilization_prediction': np.random.uniform(0.6, 0.9),
            'price_competitiveness': 1 / provider_data['price_per_hour'].values[0]
        }
        
        return performance_prediction

def main():
    # Demonstration of provider matching
    matcher = ProviderMatcher()
    
    # Register sample providers
    providers = [
        {
            'provider_id': 'provider_1',
            'cpu_cores': 16,
            'memory_gb': 64,
            'gpu_enabled': True,
            'gpu_memory': 16,
            'network_bandwidth': 1000,
            'reliability_score': 0.95,
            'price_per_hour': 0.5
        },
        {
            'provider_id': 'provider_2',
            'cpu_cores': 8,
            'memory_gb': 32,
            'gpu_enabled': False,
            'gpu_memory': 0,
            'network_bandwidth': 500,
            'reliability_score': 0.85,
            'price_per_hour': 0.3
        }
    ]
    
    for provider in providers:
        matcher.register_provider(provider)
    
    # Sample task requirements
    task_requirements = {
        'cpu_cores': 8,
        'memory_gb': 32,
        'gpu_required': False,
        'network_bandwidth': 500
    }
    
    # Find best providers
    best_providers = matcher.find_best_providers(task_requirements)
    print("Best Providers:", best_providers)
    
    # Cluster providers
    provider_clusters = matcher.cluster_providers()
    print("Provider Clusters:", provider_clusters)
    
    # Predict performance
    performance = matcher.predict_provider_performance('provider_1')
    print("Performance Prediction:", performance)

if __name__ == '__main__':
    main()
