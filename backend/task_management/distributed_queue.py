import uuid
import time
import threading
import queue
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class ComputeTask:
    """Represents a computational task in the distributed system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # Default priority
    status: str = 'PENDING'
    created_at: float = field(default_factory=time.time)
    provider_id: str = None
    compute_requirements: Dict[str, Any] = field(default_factory=lambda: {
        'cpu_cores': 1,
        'memory_gb': 4,
        'gpu_required': False
    })
    result: Any = None
    error: str = None

class DistributedTaskQueue:
    """Advanced distributed task management system."""
    
    def __init__(self, max_concurrent_tasks=10, logging_level=logging.INFO):
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Logging setup
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
        # Task processing threads
        self.processing_threads = []
        self.stop_event = threading.Event()
        
        # Metrics tracking
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_processing_time': 0
        }

    def add_task(self, task: ComputeTask):
        """Add a task to the distributed queue."""
        self.task_queue.put((-task.priority, task))
        self.metrics['total_tasks'] += 1
        self.logger.info(f"Task {task.id} added to queue")
        
        # Dynamically start processing threads if needed
        if len(self.processing_threads) < self.max_concurrent_tasks:
            self._start_processing_thread()

    def _start_processing_thread(self):
        """Start a new task processing thread."""
        thread = threading.Thread(target=self._process_tasks)
        thread.start()
        self.processing_threads.append(thread)
        self.logger.info(f"Started processing thread. Total threads: {len(self.processing_threads)}")

    def _process_tasks(self):
        """Background thread to process tasks from the queue."""
        while not self.stop_event.is_set():
            try:
                # Wait for a task with a timeout
                _, task = self.task_queue.get(timeout=5)
                
                try:
                    # Simulate task processing
                    start_time = time.time()
                    result = self._execute_task(task)
                    
                    # Update task status and metrics
                    task.status = 'COMPLETED'
                    task.result = result
                    self.completed_tasks[task.id] = task
                    self.metrics['completed_tasks'] += 1
                    
                    processing_time = time.time() - start_time
                    self._update_processing_time_metric(processing_time)
                    
                except Exception as e:
                    # Handle task failure
                    task.status = 'FAILED'
                    task.error = str(e)
                    self.failed_tasks[task.id] = task
                    self.metrics['failed_tasks'] += 1
                    self.logger.error(f"Task {task.id} failed: {e}")
                
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks, sleep briefly
                time.sleep(1)

    def _execute_task(self, task: ComputeTask):
        """Execute a computational task."""
        # Placeholder for actual task execution logic
        # In a real system, this would dispatch to appropriate compute provider
        self.logger.info(f"Executing task {task.id}")
        
        # Simulate some computation
        time.sleep(2)  # Simulated processing time
        
        return {
            'task_id': task.id,
            'result': f"Processed payload: {task.payload}",
            'timestamp': time.time()
        }

    def _update_processing_time_metric(self, new_processing_time):
        """Update average processing time metric."""
        total_completed = self.metrics['completed_tasks']
        current_avg = self.metrics['average_processing_time']
        
        if total_completed > 0:
            self.metrics['average_processing_time'] = (
                (current_avg * (total_completed - 1) + new_processing_time) 
                / total_completed
            )

    def get_task_status(self, task_id):
        """Retrieve status of a specific task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        elif task_id in self.failed_tasks:
            return self.failed_tasks[task_id]
        
        return None

    def get_system_metrics(self):
        """Retrieve current system metrics."""
        return {
            'total_tasks': self.metrics['total_tasks'],
            'completed_tasks': self.metrics['completed_tasks'],
            'failed_tasks': self.metrics['failed_tasks'],
            'average_processing_time': self.metrics['average_processing_time'],
            'active_threads': len(self.processing_threads)
        }

    def shutdown(self):
        """Gracefully shut down the task processing system."""
        self.stop_event.set()
        for thread in self.processing_threads:
            thread.join()
        self.logger.info("Task processing system shut down")

def main():
    """Demonstration of distributed task queue."""
    task_queue = DistributedTaskQueue(max_concurrent_tasks=5)
    
    # Create sample tasks
    tasks = [
        ComputeTask(payload={'data': 'task1'}, priority=3),
        ComputeTask(payload={'data': 'task2'}, priority=1),
        ComputeTask(payload={'data': 'task3'}, priority=2)
    ]
    
    # Add tasks to queue
    for task in tasks:
        task_queue.add_task(task)
    
    # Wait a bit to allow processing
    time.sleep(10)
    
    # Print system metrics
    print("System Metrics:", task_queue.get_system_metrics())
    
    # Shutdown
    task_queue.shutdown()

if __name__ == '__main__':
    main()
