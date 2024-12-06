import logging
from typing import List, Dict
import random
import time

class TaskAllocationManager:
    def __init__(self, providers, max_retries=3):
        self.providers = providers
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def allocate_task(self, task):
        """
        Allocate a task to the most suitable provider with built-in redundancy
        
        Args:
            task (Task): The computational task to be allocated
        
        Returns:
            Provider: Selected provider for task execution
        """
        suitable_providers = self._find_suitable_providers(task)
        
        if not suitable_providers:
            self.logger.error(f"No suitable providers found for task {task.id}")
            return None

        # Sort providers by reputation and availability
        ranked_providers = sorted(
            suitable_providers, 
            key=lambda p: (p.reputation, p.available_compute), 
            reverse=True
        )

        # Attempt task allocation with retry mechanism
        for attempt in range(self.max_retries):
            try:
                selected_provider = self._select_provider(ranked_providers)
                
                # Simulate task allocation with potential failure
                allocation_success = self._simulate_allocation(selected_provider, task)
                
                if allocation_success:
                    self.logger.info(f"Task {task.id} allocated to provider {selected_provider.id}")
                    return selected_provider
                
            except Exception as e:
                self.logger.warning(f"Allocation attempt {attempt + 1} failed: {e}")
                time.sleep(1)  # Wait before retry
        
        self.logger.error(f"Failed to allocate task {task.id} after {self.max_retries} attempts")
        return None

    def _find_suitable_providers(self, task):
        """
        Find providers matching task requirements
        
        Args:
            task (Task): Computational task
        
        Returns:
            List[Provider]: Providers capable of handling the task
        """
        return [
            provider for provider in self.providers
            if (provider.available_compute >= task.compute_required and
                provider.reputation >= task.minimum_reputation)
        ]

    def _select_provider(self, providers):
        """
        Select a provider using weighted random selection
        
        Args:
            providers (List[Provider]): Ranked list of providers
        
        Returns:
            Provider: Selected provider
        """
        # Implement weighted random selection
        total_weight = sum(p.reputation for p in providers)
        selection_point = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for provider in providers:
            cumulative_weight += provider.reputation
            if cumulative_weight >= selection_point:
                return provider
        
        return providers[0]  # Fallback to top provider

    def _simulate_allocation(self, provider, task, failure_probability=0.1):
        """
        Simulate task allocation with potential failure
        
        Args:
            provider (Provider): Selected provider
            task (Task): Task to be allocated
            failure_probability (float): Chance of allocation failure
        
        Returns:
            bool: Whether allocation was successful
        """
        if random.random() < failure_probability:
            raise Exception("Random allocation failure")
        
        # Reduce provider's available compute
        provider.available_compute -= task.compute_required
        
        return True

    def handle_task_failure(self, task, provider):
        """
        Handle task execution failure
        
        Args:
            task (Task): Failed task
            provider (Provider): Provider that failed the task
        """
        # Reduce provider's reputation
        provider.reputation *= 0.9
        
        # Log failure details
        self.logger.error(f"Task {task.id} failed on provider {provider.id}")
        
        # Potentially reallocate task
        self.allocate_task(task)

def main():
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Mock providers and task
    class MockProvider:
        def __init__(self, id, available_compute, reputation):
            self.id = id
            self.available_compute = available_compute
            self.reputation = reputation
    
    class MockTask:
        def __init__(self, id, compute_required, minimum_reputation):
            self.id = id
            self.compute_required = compute_required
            self.minimum_reputation = minimum_reputation
    
    providers = [
        MockProvider(1, 100, 0.9),
        MockProvider(2, 200, 0.7),
        MockProvider(3, 150, 0.8)
    ]
    
    task = MockTask(1, 50, 0.6)
    
    allocation_manager = TaskAllocationManager(providers)
    selected_provider = allocation_manager.allocate_task(task)
    
    print(f"Selected Provider: {selected_provider.id if selected_provider else 'None'}")

if __name__ == '__main__':
    main()
