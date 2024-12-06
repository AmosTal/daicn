import uuid
from enum import Enum
from typing import Dict, Any, List
from datetime import datetime, timedelta

class TaskStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    QUEUED = 'queued'

class Task:
    def __init__(self, task_type: str, payload: Dict[str, Any], priority: int = 0):
        self.id = str(uuid.uuid4())
        self.type = task_type
        self.payload = payload
        self.priority = priority
        self.status = TaskStatus.QUEUED
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.attempts = 0
        self.max_attempts = 3

class DistributedTaskQueue:
    def __init__(self):
        self.queue: List[Task] = []
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}

    def enqueue(self, task: Task):
        """Add a task to the queue with priority sorting"""
        self.queue.append(task)
        self.queue.sort(key=lambda x: x.priority, reverse=True)

    def dequeue(self) -> Task:
        """Remove and return the highest priority task"""
        if not self.queue:
            raise ValueError("Task queue is empty")
        return self.queue.pop(0)

    def get_next_task(self) -> Task:
        """Peek at the next task without removing it"""
        if not self.queue:
            raise ValueError("Task queue is empty")
        return self.queue[0]

    def mark_task_processing(self, task_id: str):
        """Mark a task as processing"""
        task = self._find_task(task_id)
        if task:
            task.status = TaskStatus.PROCESSING
            task.attempts += 1
            task.updated_at = datetime.now()

    def complete_task(self, task_id: str, result: Dict[str, Any] = None):
        """Mark a task as completed"""
        task = self._find_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.now()
            self.completed_tasks[task_id] = task
            self.queue = [t for t in self.queue if t.id != task_id]

    def fail_task(self, task_id: str, error: str = None):
        """Mark a task as failed"""
        task = self._find_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now()
            
            if task.attempts >= task.max_attempts:
                self.failed_tasks[task_id] = task
                self.queue = [t for t in self.queue if t.id != task_id]
            else:
                # Requeue the task with reduced priority
                task.priority = max(0, task.priority - 1)
                self.enqueue(task)

    def _find_task(self, task_id: str) -> Task:
        """Find a task in the queue by its ID"""
        for task in self.queue:
            if task.id == task_id:
                return task
        return None

    def get_queue_status(self) -> Dict[str, int]:
        """Get the current status of the task queue"""
        return {
            'total_tasks': len(self.queue),
            'pending_tasks': len([t for t in self.queue if t.status == TaskStatus.QUEUED]),
            'processing_tasks': len([t for t in self.queue if t.status == TaskStatus.PROCESSING]),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }

# Example usage
def main():
    task_queue = DistributedTaskQueue()

    # Create some sample tasks
    compute_task = Task('compute', {'input': 'large_dataset'}, priority=10)
    storage_task = Task('storage', {'action': 'backup'}, priority=5)
    network_task = Task('network', {'operation': 'ping'}, priority=1)

    task_queue.enqueue(compute_task)
    task_queue.enqueue(storage_task)
    task_queue.enqueue(network_task)

    print(task_queue.get_queue_status())

if __name__ == '__main__':
    main()
