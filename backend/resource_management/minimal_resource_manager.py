import psutil
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum, auto

from ..logging.minimal_logger import MinimalLogger
from ..config.minimal_config import MinimalConfigurationManager

class ResourceType(Enum):
    """Enumeration of computational resource types"""
    CPU = auto()
    MEMORY = auto()
    DISK = auto()
    NETWORK = auto()

@dataclass
class ResourceAllocation:
    """
    Representation of resource allocation for a computational task
    
    Minimal design focusing on core resource tracking
    """
    task_id: str
    resource_type: ResourceType
    allocated_percentage: float = 0.0
    max_allocation: float = 100.0
    
    start_time: float = field(default_factory=time.time)
    estimated_duration: float = 0.0
    
    def is_allocation_valid(self) -> bool:
        """
        Check if resource allocation is valid
        
        Returns:
            bool: Whether allocation is within acceptable limits
        """
        return (
            0 <= self.allocated_percentage <= self.max_allocation and
            self.estimated_duration >= 0
        )

class MinimalResourceManager:
    """
    Minimal Resource Management System
    
    Key Design Principles:
    - Simple resource tracking
    - Basic allocation mechanisms
    - Minimal system overhead
    - Predictable resource utilization
    """
    
    def __init__(
        self, 
        config: Optional[MinimalConfigurationManager] = None,
        logger: Optional[MinimalLogger] = None
    ):
        """
        Initialize resource manager
        
        Args:
            config (Optional[MinimalConfigurationManager]): Configuration manager
            logger (Optional[MinimalLogger]): Logging system
        """
        # Use provided or create default configuration
        self.config = config or MinimalConfigurationManager()
        
        # Use provided or create default logger
        self.logger = logger or MinimalLogger()
        
        # Resource allocation tracking
        self.active_allocations: Dict[str, List[ResourceAllocation]] = {}
        
        # Resource limits from configuration
        self.cpu_limit = self.config.get('resources.cpu_allocation_percentage', 60)
        self.memory_limit = self.config.get('resources.memory_allocation_percentage', 70)
    
    def get_system_resources(self) -> Dict[ResourceType, float]:
        """
        Retrieve current system resource utilization
        
        Returns:
            Dict[ResourceType, float]: Current resource utilization percentages
        """
        try:
            return {
                ResourceType.CPU: psutil.cpu_percent(),
                ResourceType.MEMORY: psutil.virtual_memory().percent,
                ResourceType.DISK: psutil.disk_usage('/').percent,
                ResourceType.NETWORK: self._get_network_usage()
            }
        except Exception as e:
            self.logger.error(
                "Failed to retrieve system resources", 
                {"error": str(e)}
            )
            return {
                ResourceType.CPU: 0,
                ResourceType.MEMORY: 0,
                ResourceType.DISK: 0,
                ResourceType.NETWORK: 0
            }
    
    def _get_network_usage(self) -> float:
        """
        Estimate network usage percentage
        
        Returns:
            float: Estimated network utilization
        """
        try:
            net_io = psutil.net_io_counters()
            # Simple network usage estimation
            return (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024 * 100)  # MB
        except Exception:
            return 0
    
    def allocate_resources(
        self, 
        task_id: str, 
        resource_type: ResourceType, 
        required_percentage: float,
        estimated_duration: float
    ) -> Dict[str, Any]:
        """
        Allocate resources for a computational task
        
        Args:
            task_id (str): Unique task identifier
            resource_type (ResourceType): Type of resource to allocate
            required_percentage (float): Percentage of resource needed
            estimated_duration (float): Estimated task duration in minutes
        
        Returns:
            Dict[str, Any]: Resource allocation result
        """
        current_resources = self.get_system_resources()
        
        # Check resource availability
        if current_resources[resource_type] + required_percentage > self._get_resource_limit(resource_type):
            self.logger.warning(
                "Resource allocation failed", 
                {
                    "task_id": task_id,
                    "resource_type": resource_type.name,
                    "current_usage": current_resources[resource_type],
                    "requested": required_percentage
                }
            )
            return {
                'status': 'error',
                'message': 'Insufficient resources'
            }
        
        # Create resource allocation
        allocation = ResourceAllocation(
            task_id=task_id,
            resource_type=resource_type,
            allocated_percentage=required_percentage,
            estimated_duration=estimated_duration
        )
        
        # Track allocation
        if task_id not in self.active_allocations:
            self.active_allocations[task_id] = []
        
        self.active_allocations[task_id].append(allocation)
        
        self.logger.info(
            "Resources allocated", 
            {
                "task_id": task_id,
                "resource_type": resource_type.name,
                "allocation_percentage": required_percentage
            }
        )
        
        return {
            'status': 'success',
            'allocation': allocation
        }
    
    def _get_resource_limit(self, resource_type: ResourceType) -> float:
        """
        Get resource allocation limit
        
        Args:
            resource_type (ResourceType): Type of resource
        
        Returns:
            float: Maximum allowed resource allocation
        """
        limits = {
            ResourceType.CPU: self.cpu_limit,
            ResourceType.MEMORY: self.memory_limit,
            ResourceType.DISK: 90,  # Conservative disk usage limit
            ResourceType.NETWORK: 90  # Conservative network usage limit
        }
        return limits.get(resource_type, 80)
    
    def release_resources(self, task_id: str):
        """
        Release resources for a completed task
        
        Args:
            task_id (str): Unique task identifier
        """
        if task_id in self.active_allocations:
            del self.active_allocations[task_id]
            
            self.logger.info(
                "Resources released", 
                {"task_id": task_id}
            )
    
    def get_task_resource_allocations(
        self, 
        task_id: str
    ) -> List[ResourceAllocation]:
        """
        Retrieve resource allocations for a specific task
        
        Args:
            task_id (str): Unique task identifier
        
        Returns:
            List[ResourceAllocation]: List of resource allocations
        """
        return self.active_allocations.get(task_id, [])

async def main():
    """
    Demonstration of Minimal Resource Manager
    """
    # Initialize resource manager
    resource_manager = MinimalResourceManager()
    
    # Display current system resources
    print("Current System Resources:")
    print(resource_manager.get_system_resources())
    
    # Simulate resource allocation for multiple tasks
    tasks = [
        ('task1', ResourceType.CPU, 20, 3),
        ('task2', ResourceType.MEMORY, 30, 5),
        ('task3', ResourceType.NETWORK, 15, 2)
    ]
    
    for task_id, resource_type, percentage, duration in tasks:
        allocation = resource_manager.allocate_resources(
            task_id, resource_type, percentage, duration
        )
        print(f"Allocation for {task_id}: {allocation}")
        
        # Simulate task processing
        await asyncio.sleep(1)
    
    # Release resources
    for task_id, _, _, _ in tasks:
        resource_manager.release_resources(task_id)

if __name__ == '__main__':
    asyncio.run(main())
