import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler

class ResourceAllocationOptimizer:
    """
    Advanced resource allocation optimizer for distributed AI computation network
    """
    
    def __init__(self):
        """
        Initialize Resource Allocation Optimizer
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Scaling utility for normalizing different resource metrics
        self.scaler = MinMaxScaler()
        
        self.logger.info("Resource Allocation Optimizer initialized")

    def compute_provider_compatibility_matrix(
        self, 
        tasks: List[Dict[str, Any]], 
        providers: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute compatibility matrix between tasks and providers
        
        Args:
            tasks (List[Dict]): List of tasks requiring computation
            providers (List[Dict]): List of available compute providers
        
        Returns:
            np.ndarray: Compatibility matrix representing task-provider matching scores
        """
        try:
            # Extract relevant features
            task_features = pd.DataFrame(tasks)
            provider_features = pd.DataFrame(providers)
            
            # Normalize features
            task_features_scaled = self.scaler.fit_transform(task_features[['complexity', 'compute_required']])
            provider_features_scaled = self.scaler.fit_transform(provider_features[['computational_power', 'reliability']])
            
            # Compute compatibility matrix
            compatibility_matrix = np.zeros((len(tasks), len(providers)))
            
            for i, task in enumerate(task_features_scaled):
                for j, provider in enumerate(provider_features_scaled):
                    # Compute compatibility score using weighted distance
                    compatibility_score = np.exp(-np.linalg.norm(task - provider))
                    compatibility_matrix[i, j] = compatibility_score
            
            return compatibility_matrix
        
        except Exception as e:
            self.logger.error(f"Compatibility matrix computation error: {e}")
            raise

    def optimize_task_allocation(
        self, 
        tasks: List[Dict[str, Any]], 
        providers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize task allocation across compute providers
        
        Args:
            tasks (List[Dict]): List of tasks requiring computation
            providers (List[Dict]): List of available compute providers
        
        Returns:
            List[Dict]: Optimized task allocation plan
        """
        try:
            # Compute compatibility matrix
            compatibility_matrix = self.compute_provider_compatibility_matrix(tasks, providers)
            
            # Use Hungarian algorithm for optimal task-provider matching
            task_indices, provider_indices = linear_sum_assignment(
                compatibility_matrix, 
                maximize=True
            )
            
            # Generate allocation plan
            allocation_plan = []
            for task_idx, provider_idx in zip(task_indices, provider_indices):
                allocation_plan.append({
                    'task_id': tasks[task_idx]['task_id'],
                    'provider_id': providers[provider_idx]['provider_id'],
                    'compatibility_score': compatibility_matrix[task_idx, provider_idx]
                })
            
            self.logger.info(f"Optimized task allocation completed. {len(allocation_plan)} tasks allocated.")
            return allocation_plan
        
        except Exception as e:
            self.logger.error(f"Task allocation optimization error: {e}")
            raise

    def simulate_dynamic_allocation(
        self, 
        tasks: List[Dict[str, Any]], 
        providers: List[Dict[str, Any]], 
        time_window: int = 60
    ) -> Dict[str, Any]:
        """
        Simulate dynamic resource allocation over a time window
        
        Args:
            tasks (List[Dict]): List of tasks requiring computation
            providers (List[Dict]): List of available compute providers
            time_window (int): Time window for allocation simulation (minutes)
        
        Returns:
            Dict: Simulation results and performance metrics
        """
        try:
            # Initial allocation
            initial_allocation = self.optimize_task_allocation(tasks, providers)
            
            # Track allocation performance
            performance_metrics = {
                'total_tasks': len(tasks),
                'allocated_tasks': len(initial_allocation),
                'allocation_efficiency': len(initial_allocation) / len(tasks),
                'average_compatibility_score': np.mean([
                    alloc['compatibility_score'] for alloc in initial_allocation
                ])
            }
            
            # Simulate dynamic changes
            dynamic_events = self._generate_dynamic_events(providers, time_window)
            
            # Update allocation based on dynamic events
            updated_providers = self._apply_dynamic_events(providers, dynamic_events)
            updated_allocation = self.optimize_task_allocation(tasks, updated_providers)
            
            performance_metrics.update({
                'dynamic_allocation_efficiency': len(updated_allocation) / len(tasks),
                'dynamic_average_compatibility_score': np.mean([
                    alloc['compatibility_score'] for alloc in updated_allocation
                ])
            })
            
            self.logger.info("Dynamic resource allocation simulation completed")
            return performance_metrics
        
        except Exception as e:
            self.logger.error(f"Dynamic allocation simulation error: {e}")
            raise

    def _generate_dynamic_events(
        self, 
        providers: List[Dict[str, Any]], 
        time_window: int
    ) -> List[Dict[str, Any]]:
        """
        Generate simulated dynamic events affecting provider resources
        
        Args:
            providers (List[Dict]): List of compute providers
            time_window (int): Time window for events
        
        Returns:
            List[Dict]: Dynamic events affecting providers
        """
        np.random.seed(42)
        
        dynamic_events = []
        for provider in providers:
            # Simulate random events: resource fluctuations, temporary unavailability
            event_probability = np.random.random()
            
            if event_probability < 0.3:  # 30% chance of an event
                event = {
                    'provider_id': provider['provider_id'],
                    'event_type': np.random.choice(['resource_reduction', 'temporary_unavailable']),
                    'impact_magnitude': np.random.uniform(0.1, 0.5),
                    'duration': np.random.randint(5, time_window)
                }
                dynamic_events.append(event)
        
        return dynamic_events

    def _apply_dynamic_events(
        self, 
        providers: List[Dict[str, Any]], 
        dynamic_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply dynamic events to provider resources
        
        Args:
            providers (List[Dict]): List of compute providers
            dynamic_events (List[Dict]): Dynamic events to apply
        
        Returns:
            List[Dict]: Updated providers after applying events
        """
        updated_providers = providers.copy()
        
        for event in dynamic_events:
            for provider in updated_providers:
                if provider['provider_id'] == event['provider_id']:
                    if event['event_type'] == 'resource_reduction':
                        provider['computational_power'] *= (1 - event['impact_magnitude'])
                    elif event['event_type'] == 'temporary_unavailable':
                        provider['reliability'] *= (1 - event['impact_magnitude'])
        
        return updated_providers

def generate_synthetic_data(num_tasks=50, num_providers=20):
    """
    Generate synthetic tasks and providers for testing
    
    Args:
        num_tasks (int): Number of tasks to generate
        num_providers (int): Number of providers to generate
    
    Returns:
        Tuple of tasks and providers lists
    """
    np.random.seed(42)
    
    tasks = [
        {
            'task_id': f'task_{i}',
            'complexity': np.random.uniform(0.1, 1),
            'compute_required': np.random.uniform(1, 100)
        }
        for i in range(num_tasks)
    ]
    
    providers = [
        {
            'provider_id': f'provider_{i}',
            'computational_power': np.random.uniform(10, 100),
            'reliability': np.random.uniform(0.5, 1)
        }
        for i in range(num_providers)
    ]
    
    return tasks, providers

def main():
    # Generate synthetic data
    tasks, providers = generate_synthetic_data()
    
    # Initialize resource allocation optimizer
    optimizer = ResourceAllocationOptimizer()
    
    # Optimize task allocation
    allocation_plan = optimizer.optimize_task_allocation(tasks, providers)
    print("Task Allocation Plan:")
    for allocation in allocation_plan:
        print(allocation)
    
    # Simulate dynamic allocation
    simulation_results = optimizer.simulate_dynamic_allocation(tasks, providers)
    print("\nDynamic Allocation Simulation Results:")
    for metric, value in simulation_results.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    main()
