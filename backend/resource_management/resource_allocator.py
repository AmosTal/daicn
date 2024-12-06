import uuid
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum, auto

class ResourceType(Enum):
    """Enumeration of computational resource types"""
    CPU = auto()
    GPU = auto()
    MEMORY = auto()
    STORAGE = auto()
    NETWORK_BANDWIDTH = auto()

class ResourceState(Enum):
    """Current state of a computational resource"""
    AVAILABLE = auto()
    ALLOCATED = auto()
    OVERLOADED = auto()
    MAINTENANCE = auto()

class ComputeProvider:
    """
    Represents a computational resource provider in the network
    """
    def __init__(
        self, 
        provider_id: str, 
        capabilities: Dict[ResourceType, float],
        location: Optional[str] = None
    ):
        """
        Initialize a compute provider
        
        Args:
            provider_id (str): Unique identifier for the provider
            capabilities (Dict): Available computational resources
            location (str, optional): Geographic or network location
        """
        self.id = provider_id
        self.capabilities = capabilities
        self.location = location
        self.current_load: Dict[ResourceType, float] = {
            resource: 0.0 for resource in ResourceType
        }
        self.state = ResourceState.AVAILABLE
        self.last_update = time.time()

    def update_load(self, resource_type: ResourceType, load: float):
        """
        Update the current load for a specific resource type
        
        Args:
            resource_type (ResourceType): Type of resource
            load (float): Current load percentage
        """
        self.current_load[resource_type] = load
        self.last_update = time.time()
        
        # Automatically update state based on load
        if load > 0.9:  # 90% load threshold
            self.state = ResourceState.OVERLOADED
        elif load > 0.7:  # 70% load threshold
            self.state = ResourceState.ALLOCATED
        else:
            self.state = ResourceState.AVAILABLE

class ResourceAllocationOptimizer:
    """
    Advanced resource allocation and optimization system
    """
    def __init__(
        self, 
        providers: Optional[List[ComputeProvider]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize Resource Allocation Optimizer
        
        Args:
            providers (List[ComputeProvider], optional): Initial list of providers
            log_level (int): Logging level
        """
        # Logging configuration
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Provider management
        self.providers: Dict[str, ComputeProvider] = {}
        if providers:
            for provider in providers:
                self.register_provider(provider)
        
        # Allocation tracking
        self.active_allocations: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time': 0
        }

    def register_provider(self, provider: ComputeProvider):
        """
        Register a new compute provider
        
        Args:
            provider (ComputeProvider): Compute provider to register
        """
        self.providers[provider.id] = provider
        self.logger.info(f"Registered provider: {provider.id}")

    async def allocate_resources(
        self, 
        task_requirements: Dict[ResourceType, float],
        allocation_strategy: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Allocate computational resources for a task
        
        Args:
            task_requirements (Dict): Required resources for the task
            allocation_strategy (str): Strategy for resource allocation
        
        Returns:
            Dict: Allocation details
        """
        start_time = time.time()
        allocation_id = str(uuid.uuid4())
        
        try:
            # Find suitable providers
            candidate_providers = self._find_suitable_providers(task_requirements)
            
            if not candidate_providers:
                self.logger.warning("No suitable providers found")
                self.performance_metrics['failed_allocations'] += 1
                return {
                    'status': 'failed',
                    'reason': 'No suitable providers'
                }
            
            # Select provider based on strategy
            selected_provider = self._select_provider(
                candidate_providers, 
                task_requirements, 
                allocation_strategy
            )
            
            # Allocate resources
            allocation_details = {
                'id': allocation_id,
                'provider_id': selected_provider.id,
                'task_requirements': task_requirements,
                'timestamp': time.time()
            }
            
            self.active_allocations[allocation_id] = allocation_details
            selected_provider.update_load(
                list(task_requirements.keys())[0], 
                task_requirements[list(task_requirements.keys())[0]]
            )
            
            # Update performance metrics
            self.performance_metrics['total_allocations'] += 1
            self.performance_metrics['successful_allocations'] += 1
            
            allocation_time = time.time() - start_time
            self.performance_metrics['average_allocation_time'] = (
                (self.performance_metrics['average_allocation_time'] * 
                 (self.performance_metrics['successful_allocations'] - 1) + 
                 allocation_time) / 
                self.performance_metrics['successful_allocations']
            )
            
            self.logger.info(f"Resources allocated: {allocation_id}")
            
            return {
                'status': 'success',
                'allocation_id': allocation_id,
                'provider': selected_provider.id
            }
        
        except Exception as e:
            self.logger.error(f"Resource allocation error: {e}")
            self.performance_metrics['failed_allocations'] += 1
            return {
                'status': 'failed',
                'reason': str(e)
            }

    def _find_suitable_providers(
        self, 
        task_requirements: Dict[ResourceType, float]
    ) -> List[ComputeProvider]:
        """
        Find providers that can meet task requirements
        
        Args:
            task_requirements (Dict): Required resources
        
        Returns:
            List[ComputeProvider]: Suitable providers
        """
        suitable_providers = []
        
        for provider in self.providers.values():
            # Check if provider can meet requirements
            if all(
                provider.capabilities.get(resource, 0) >= requirement and
                provider.current_load.get(resource, 0) + requirement <= 1.0
                for resource, requirement in task_requirements.items()
            ):
                suitable_providers.append(provider)
        
        return suitable_providers

    def _select_provider(
        self, 
        candidates: List[ComputeProvider], 
        requirements: Dict[ResourceType, float],
        strategy: str = 'balanced'
    ) -> ComputeProvider:
        """
        Select the most appropriate provider based on strategy
        
        Args:
            candidates (List[ComputeProvider]): Potential providers
            requirements (Dict): Task resource requirements
            strategy (str): Selection strategy
        
        Returns:
            ComputeProvider: Selected provider
        """
        if strategy == 'lowest_load':
            return min(
                candidates, 
                key=lambda p: sum(p.current_load.values())
            )
        elif strategy == 'geographic_proximity':
            # Placeholder for more complex geographic selection
            return candidates[0]
        else:  # balanced strategy
            return min(
                candidates, 
                key=lambda p: abs(
                    sum(p.current_load.values()) / len(p.current_load) - 0.5
                )
            )

    def release_resources(self, allocation_id: str):
        """
        Release resources for a completed allocation
        
        Args:
            allocation_id (str): Unique allocation identifier
        """
        if allocation_id not in self.active_allocations:
            self.logger.warning(f"Unknown allocation: {allocation_id}")
            return
        
        allocation = self.active_allocations[allocation_id]
        provider = self.providers.get(allocation['provider_id'])
        
        if provider:
            # Reset provider load
            for resource in allocation['task_requirements']:
                provider.update_load(resource, 0)
        
        del self.active_allocations[allocation_id]
        self.logger.info(f"Resources released: {allocation_id}")

    def get_system_overview(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive system resource overview
        
        Returns:
            Dict: System resource status and performance metrics
        """
        return {
            'providers': {
                pid: {
                    'capabilities': p.capabilities,
                    'current_load': p.current_load,
                    'state': p.state.name
                } for pid, p in self.providers.items()
            },
            'performance_metrics': self.performance_metrics,
            'active_allocations': len(self.active_allocations)
        }

def main():
    """
    Demonstration of resource allocation system
    """
    async def example_usage():
        # Create compute providers
        providers = [
            ComputeProvider(
                'provider_1', 
                {
                    ResourceType.CPU: 1.0, 
                    ResourceType.MEMORY: 1.0, 
                    ResourceType.GPU: 0.5
                }
            ),
            ComputeProvider(
                'provider_2', 
                {
                    ResourceType.CPU: 0.8, 
                    ResourceType.MEMORY: 0.9, 
                    ResourceType.GPU: 0.7
                }
            )
        ]
        
        # Initialize resource allocation optimizer
        allocator = ResourceAllocationOptimizer(providers)
        
        # Allocate resources for a task
        task_requirements = {
            ResourceType.CPU: 0.3,
            ResourceType.MEMORY: 0.4
        }
        
        allocation_result = await allocator.allocate_resources(task_requirements)
        print("Allocation Result:", allocation_result)
        
        # Get system overview
        system_overview = allocator.get_system_overview()
        print("System Overview:", system_overview)
    
    # Run the example
    asyncio.run(example_usage())

if __name__ == '__main__':
    main()
