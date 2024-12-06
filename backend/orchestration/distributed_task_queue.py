import uuid
import time
import threading
import queue
import logging
from typing import Dict, Any, List, Callable
import redis
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

class DistributedTaskQueue:
    """
    Advanced Distributed Task Queue for Decentralized AI Computation Network
    
    Provides robust, scalable task distribution and management
    """
    
    def __init__(
        self, 
        redis_host: str = 'localhost', 
        redis_port: int = 6379,
        max_workers: int = None
    ):
        """
        Initialize Distributed Task Queue
        
        Args:
            redis_host (str): Redis server host
            redis_port (int): Redis server port
            max_workers (int): Maximum number of concurrent workers
        """
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Redis connection for distributed queue
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=True
            )
            self.redis_client.ping()
            self.logger.info("Redis connection established successfully")
        except Exception as e:
            self.logger.error(f"Redis connection error: {e}")
            raise
        
        # Task queue configuration
        self.task_queue_name = 'daicn:task_queue'
        self.result_queue_name = 'daicn:result_queue'
        
        # Concurrency management
        self.max_workers = max_workers or (multiprocessing.cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Task tracking
        self.active_tasks = {}
        self.task_lock = threading.Lock()
        
        self.logger.info(f"Distributed Task Queue initialized with {self.max_workers} workers")

    def enqueue_task(
        self, 
        task_function: Callable, 
        task_args: List[Any] = None, 
        task_kwargs: Dict[str, Any] = None,
        priority: int = 5
    ) -> str:
        """
        Enqueue a task for distributed processing
        
        Args:
            task_function (Callable): Function to execute
            task_args (List[Any]): Positional arguments for task
            task_kwargs (Dict[str, Any]): Keyword arguments for task
            priority (int): Task priority (1-10, 1 being highest)
        
        Returns:
            str: Unique task ID
        """
        task_args = task_args or []
        task_kwargs = task_kwargs or {}
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Prepare task payload
        task_payload = {
            'task_id': task_id,
            'function_name': task_function.__name__,
            'module_name': task_function.__module__,
            'args': task_args,
            'kwargs': task_kwargs,
            'priority': priority,
            'timestamp': time.time(),
            'status': 'pending'
        }
        
        try:
            # Store task in Redis queue with priority
            self.redis_client.zadd(
                self.task_queue_name, 
                {json.dumps(task_payload): -priority}
            )
            
            # Track active task
            with self.task_lock:
                self.active_tasks[task_id] = task_payload
            
            self.logger.info(f"Task {task_id} enqueued successfully")
            return task_id
        
        except Exception as e:
            self.logger.error(f"Task enqueue error: {e}")
            raise

    def process_tasks(self, worker_id: str = None):
        """
        Continuously process tasks from distributed queue
        
        Args:
            worker_id (str): Optional unique worker identifier
        """
        worker_id = worker_id or str(uuid.uuid4())
        
        while True:
            try:
                # Retrieve next task from queue
                task_data = self._get_next_task()
                
                if task_data is None:
                    time.sleep(1)  # Prevent tight loop
                    continue
                
                # Execute task
                result = self._execute_task(task_data, worker_id)
                
                # Store result
                self._store_task_result(task_data, result)
            
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")

    def _get_next_task(self) -> Dict[str, Any]:
        """
        Retrieve next task from Redis queue
        
        Returns:
            Dict[str, Any]: Task payload or None
        """
        try:
            # Retrieve and remove highest priority task
            task_entries = self.redis_client.zrange(
                self.task_queue_name, 0, 0, withscores=True
            )
            
            if not task_entries:
                return None
            
            task_payload, _ = task_entries[0]
            task_data = json.loads(task_payload)
            
            # Remove task from queue
            self.redis_client.zrem(self.task_queue_name, task_payload)
            
            return task_data
        
        except Exception as e:
            self.logger.error(f"Task retrieval error: {e}")
            raise

    def _execute_task(
        self, 
        task_data: Dict[str, Any], 
        worker_id: str
    ) -> Any:
        """
        Execute a task and track its progress
        
        Args:
            task_data (Dict[str, Any]): Task payload
            worker_id (str): Worker identifier
        
        Returns:
            Task execution result
        """
        try:
            # Import task function dynamically
            module = __import__(task_data['module_name'], fromlist=[task_data['function_name']])
            task_function = getattr(module, task_data['function_name'])
            
            # Update task status
            task_data['status'] = 'running'
            task_data['worker_id'] = worker_id
            
            # Execute task
            result = task_function(*task_data['args'], **task_data['kwargs'])
            
            task_data['status'] = 'completed'
            return result
        
        except Exception as e:
            task_data['status'] = 'failed'
            task_data['error'] = str(e)
            self.logger.error(f"Task execution error: {e}")
            raise

    def _store_task_result(
        self, 
        task_data: Dict[str, Any], 
        result: Any
    ):
        """
        Store task result in Redis result queue
        
        Args:
            task_data (Dict[str, Any]): Original task payload
            result (Any): Task execution result
        """
        try:
            result_payload = {
                **task_data,
                'result': result,
                'completion_time': time.time()
            }
            
            # Store result in Redis
            self.redis_client.rpush(
                self.result_queue_name, 
                json.dumps(result_payload)
            )
            
            self.logger.info(f"Task {task_data['task_id']} result stored")
        
        except Exception as e:
            self.logger.error(f"Result storage error: {e}")
            raise

    def get_task_results(self, timeout: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve task results from result queue
        
        Args:
            timeout (int): Blocking timeout in seconds
        
        Returns:
            List of task results
        """
        try:
            # Retrieve results from Redis
            results = self.redis_client.blpop(
                self.result_queue_name, 
                timeout=timeout
            )
            
            if not results:
                return []
            
            # Parse results
            return [json.loads(results[1])]
        
        except Exception as e:
            self.logger.error(f"Result retrieval error: {e}")
            raise

    def start_workers(self, num_workers: int = None):
        """
        Start multiple worker threads
        
        Args:
            num_workers (int): Number of workers to start
        """
        num_workers = num_workers or self.max_workers
        
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self.process_tasks, 
                daemon=True
            )
            worker.start()
            workers.append(worker)
        
        return workers

def example_task(x: int, y: int) -> int:
    """
    Example task function for demonstration
    
    Args:
        x (int): First number
        y (int): Second number
    
    Returns:
        int: Sum of x and y
    """
    time.sleep(1)  # Simulate computation
    return x + y

def main():
    # Initialize distributed task queue
    task_queue = DistributedTaskQueue()
    
    # Start worker threads
    task_queue.start_workers(num_workers=4)
    
    # Enqueue multiple tasks
    task_ids = [
        task_queue.enqueue_task(example_task, [i, i+1]) 
        for i in range(10)
    ]
    
    # Retrieve and print results
    while task_ids:
        results = task_queue.get_task_results(timeout=5)
        
        for result in results:
            print(f"Task {result['task_id']} Result: {result['result']}")
            task_ids.remove(result['task_id'])

if __name__ == '__main__':
    main()
