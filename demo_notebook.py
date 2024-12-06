import sys
import os
import asyncio

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath('../'))

from backend.ml.task_predictor import TaskPredictor
from backend.security.basic_auth import AuthenticationManager
from backend.task_queue.minimal_task_queue import MinimalTaskQueue
from backend.resource_management.minimal_resource_manager import MinimalResourceManager
from backend.error_handling.error_manager import ErrorManager, ErrorCategory

async def main():
    """
    DAICN MVP Demonstration Script
    """
    print("DAICN: Decentralized AI Computation Network - MVP Demonstration\n")

    # Initialize components
    task_predictor = TaskPredictor()
    task_queue = MinimalTaskQueue()
    resource_manager = MinimalResourceManager()
    auth_manager = AuthenticationManager()
    error_manager = ErrorManager()

    # 1. Task Complexity Prediction
    print("=== Task Complexity Prediction ===")
    tasks = [
        {"type": "image_classification", "data_size": 1000},
        {"type": "natural_language_processing", "data_size": 5000},
        {"type": "time_series_analysis", "data_size": 2500}
    ]

    for task in tasks:
        complexity = task_predictor.predict_complexity(task)
        resource_req = resource_manager.estimate_resources(complexity)
        
        print(f"Task: {task['type']}")
        print(f"Complexity: {complexity}")
        print(f"Resource Requirements: {resource_req}\n")

    # 2. Authentication Demonstration
    print("=== Authentication Demonstration ===")
    auth_manager.register_user("demo_user", "password123", role="USER")
    auth_manager.register_user("demo_provider", "provider456", role="PROVIDER")

    def test_authentication(username, password):
        try:
            user = auth_manager.authenticate_user(username, password)
            print(f"Authentication Successful for {username}")
            print(f"User Role: {user['role']}")
        except Exception as e:
            print(f"Authentication Failed: {e}")

    test_authentication("demo_user", "password123")
    test_authentication("demo_provider", "wrong_password")

    # 3. Error Handling Demonstration
    print("\n=== Error Handling Demonstration ===")
    async def demonstrate_error_handling():
        # Authentication Error
        try:
            raise PermissionError("Invalid credentials")
        except Exception as e:
            result = await error_manager.handle_error(e)
            print("Authentication Error Handling:", result)
        
        # Resource Allocation Error
        try:
            resource_manager.allocate_resources(
                task_id='overload_test', 
                resource_type='CPU', 
                required_percentage=200
            )
        except Exception as e:
            result = await error_manager.handle_error(e)
            print("Resource Allocation Error Handling:", result)

    await demonstrate_error_handling()

if __name__ == "__main__":
    asyncio.run(main())
