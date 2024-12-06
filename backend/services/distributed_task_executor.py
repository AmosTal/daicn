import asyncio
import logging
from typing import Dict, Any, List

from .task_processing import TaskProcessingService

class DistributedTaskExecutor:
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.task_processing_service = TaskProcessingService()
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()

    async def submit_task(self, task_data: Dict[str, Any]):
        """
        Submit a task to the distributed task queue
        """
        await self.task_queue.put(task_data)
        self.logger.info(f"Task submitted: {task_data.get('task_type', 'unknown')}")

    async def worker(self, worker_id: int):
        """
        Worker method to process tasks from the queue
        """
        while True:
            try:
                # Wait for a task from the queue
                task_data = await self.task_queue.get()
                
                self.logger.info(f"Worker {worker_id} processing task")
                
                # Process the task
                try:
                    result = await self.task_processing_service.process_ai_task(task_data)
                    await self.results_queue.put({
                        'task_data': task_data,
                        'result': result
                    })
                except Exception as e:
                    self.logger.error(f"Task processing error: {e}")
                    await self.results_queue.put({
                        'task_data': task_data,
                        'result': {
                            'status': 'failed',
                            'error': str(e)
                        }
                    })
                
                # Mark task as done
                self.task_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in worker {worker_id}: {e}")

    async def result_collector(self):
        """
        Collect and process results from the results queue
        """
        while True:
            try:
                result_data = await self.results_queue.get()
                task_data = result_data['task_data']
                result = result_data['result']
                
                # Log result or perform additional processing
                if result['status'] == 'completed':
                    self.logger.info(f"Task completed: {task_data.get('task_type', 'unknown')}")
                else:
                    self.logger.warning(f"Task failed: {result.get('error', 'Unknown error')}")
                
                self.results_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in result collector: {e}")

    async def start(self):
        """
        Start the distributed task executor
        """
        # Create worker pool
        workers = [
            asyncio.create_task(self.worker(i)) 
            for i in range(self.max_workers)
        ]
        
        # Start result collector
        result_collector = asyncio.create_task(self.result_collector())
        
        self.logger.info(f"Distributed task executor started with {self.max_workers} workers")
        
        return workers + [result_collector]

    async def stop(self, workers):
        """
        Gracefully stop the distributed task executor
        """
        for worker in workers:
            worker.cancel()
        
        # Wait for all tasks to complete
        await self.task_queue.join()
        await self.results_queue.join()
        
        self.logger.info("Distributed task executor stopped")

    async def execute_tasks(self, tasks: List[Dict[str, Any]]):
        """
        Execute a batch of tasks
        """
        # Submit all tasks to the queue
        for task in tasks:
            await self.submit_task(task)
        
        # Wait for all tasks to complete
        await self.task_queue.join()
        await self.results_queue.join()

    @classmethod
    async def run_example(cls):
        """
        Example usage of the distributed task executor
        """
        executor = cls(max_workers=4)
        
        # Example tasks
        tasks = [
            {
                'framework': 'pytorch',
                'task_type': 'training',
                'model_config': {
                    'type': 'linear',
                    'input_size': 10,
                    'output_size': 1
                },
                'input_data': [[float(i) for i in range(10)] for _ in range(100)],
                'hyperparameters': {
                    'learning_rate': 0.001,
                    'epochs': 5
                }
            },
            {
                'framework': 'tensorflow',
                'task_type': 'inference',
                'model_config': {
                    'type': 'sequential',
                    'input_size': 10,
                    'output_size': 1
                },
                'input_data': [[float(i) for i in range(10)] for _ in range(50)]
            }
        ]
        
        try:
            # Start the executor
            workers = await executor.start()
            
            # Execute tasks
            await executor.execute_tasks(tasks)
            
            # Stop the executor
            await executor.stop(workers)
        
        except Exception as e:
            print(f"Error in example run: {e}")
