import uuid
import asyncio
from enum import Enum, auto
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..logging.minimal_logger import MinimalLogger
from ..ml.task_predictor import MLTaskPredictor, TaskComplexityLevel

class TaskStatus(Enum):
    """Possible states of a computational task"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    QUEUED = auto()

@dataclass
class ComputationalTask:
    """
    Representation of a computational task in the system
    
    Minimal design focusing on core task management
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    task_type: str = 'generic'
    status: TaskStatus = TaskStatus.PENDING
    
    # Task characteristics
    compute_intensity: float = 0.5
    memory_requirement: float = 0.5
    data_volume: float = 0.0
    priority: int = 3
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    estimated_completion_time: Optional[datetime] = None
    
    def __post_init__(self):
        """
        Post-initialization setup
        """
        self.estimated_completion_time = self.created_at + timedelta(
            minutes=self._estimate_completion_time()
        )
    
    def _estimate_completion_time(self) -> float:
        """
        Estimate task completion time based on characteristics
        
        Returns:
            float: Estimated completion time in minutes
        """
        base_time = 5.0  # Base processing time
        complexity_multiplier = {
            TaskComplexityLevel.LOW.name: 1.0,
            TaskComplexityLevel.MEDIUM.name: 2.0,
            TaskComplexityLevel.HIGH.name: 4.0,
            TaskComplexityLevel.CRITICAL.name: 8.0
        }
        
        return base_time * (
            1 + self.compute_intensity * 0.5 +
            self.memory_requirement * 0.3 +
            complexity_multiplier.get(
                MLTaskPredictor().predict_task_complexity(
                    {
                        'compute_intensity': self.compute_intensity,
                        'memory_requirement': self.memory_requirement,
                        'data_volume': self.data_volume,
                        'priority': self.priority
                    }
                )['complexity_level'], 
                1.0
            )
        )

class MinimalTaskQueue:
    """
    Minimal Task Queue for MVP
    
    Key Design Principles:
    - Simple task management
    - Basic queuing mechanism
    - Minimal resource tracking
    """
    
    def __init__(
        self, 
        max_concurrent_tasks: int = 10,
        log_level: int = None
    ):
        """
        Initialize task queue
        
        Args:
            max_concurrent_tasks (int): Maximum number of tasks processed simultaneously
            log_level (Optional[int]): Logging level
        """
        self.tasks: Dict[str, ComputationalTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize logger
        self.logger = MinimalLogger(log_level=log_level)
        
        # Task processing semaphore
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def submit_task(
        self, 
        task: ComputationalTask
    ) -> Dict[str, Any]:
        """
        Submit a new computational task
        
        Args:
            task (ComputationalTask): Task to be submitted
        
        Returns:
            Dict[str, Any]: Task submission result
        """
        # Check queue capacity
        if len(self.tasks) >= self.max_concurrent_tasks:
            self.logger.warning(
                "Task queue at maximum capacity", 
                {"current_tasks": len(self.tasks)}
            )
            return {
                'status': 'error',
                'message': 'Task queue full'
            }
        
        # Add task to tracking
        self.tasks[task.task_id] = task
        await self.task_queue.put(task)
        
        self.logger.info(
            "Task submitted", 
            {
                "task_id": task.task_id, 
                "task_type": task.task_type
            }
        )
        
        return {
            'status': 'success',
            'task_id': task.task_id
        }
    
    async def process_tasks(self):
        """
        Continuously process tasks from the queue
        """
        while True:
            task = await self.task_queue.get()
            
            async with self.task_semaphore:
                try:
                    # Simulate task processing
                    await self._process_task(task)
                except Exception as e:
                    self.logger.error(
                        "Task processing failed", 
                        {
                            "task_id": task.task_id, 
                            "error": str(e)
                        }
                    )
                    task.status = TaskStatus.FAILED
                finally:
                    self.task_queue.task_done()
    
    async def _process_task(self, task: ComputationalTask):
        """
        Process an individual task
        
        Args:
            task (ComputationalTask): Task to process
        """
        task.status = TaskStatus.PROCESSING
        task.updated_at = datetime.now()
        
        # Simulate task processing time
        processing_time = task._estimate_completion_time()
        await asyncio.sleep(processing_time / 10)  # Scaled down for demonstration
        
        task.status = TaskStatus.COMPLETED
        task.updated_at = datetime.now()
        
        self.logger.info(
            "Task completed", 
            {
                "task_id": task.task_id, 
                "processing_time": processing_time
            }
        )
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Retrieve task status
        
        Args:
            task_id (str): Unique task identifier
        
        Returns:
            Optional[TaskStatus]: Current task status
        """
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def list_tasks(
        self, 
        status: Optional[TaskStatus] = None
    ) -> List[ComputationalTask]:
        """
        List tasks, optionally filtered by status
        
        Args:
            status (Optional[TaskStatus]): Filter tasks by status
        
        Returns:
            List[ComputationalTask]: Matching tasks
        """
        if status:
            return [
                task for task in self.tasks.values() 
                if task.status == status
            ]
        return list(self.tasks.values())

async def main():
    """
    Demonstration of Minimal Task Queue
    """
    task_queue = MinimalTaskQueue()
    
    # Create some sample tasks
    tasks = [
        ComputationalTask(
            task_type='machine_learning_training',
            compute_intensity=0.8,
            memory_requirement=0.7,
            priority=4
        ) for _ in range(5)
    ]
    
    # Submit tasks
    for task in tasks:
        await task_queue.submit_task(task)
    
    # Start task processing
    processing_task = asyncio.create_task(task_queue.process_tasks())
    
    # Wait for tasks to complete
    await task_queue.task_queue.join()
    
    # Cancel processing task
    processing_task.cancel()
    
    # Print final task statuses
    for task in task_queue.list_tasks():
        print(f"Task {task.task_id}: {task.status.name}")

if __name__ == '__main__':
    asyncio.run(main())
