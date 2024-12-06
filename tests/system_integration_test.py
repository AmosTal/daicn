import asyncio
import unittest
import sys
import os
import traceback

# Print current working directory and Python path
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print("Updated Python Path:", sys.path)

# Verbose import attempts
def verbose_import(module_path):
    try:
        module = __import__(module_path, fromlist=[''])
        print(f"Successfully imported {module_path}")
        # Return the specific class instead of the module
        if module_path == 'backend.task_queue.task_queue':
            return module.DistributedTaskQueue
        elif module_path == 'backend.resource_management.resource_allocator':
            return module.ResourceAllocationOptimizer
        elif module_path == 'backend.ml.task_predictor':
            return module.MLTaskPredictor
        elif module_path == 'backend.security.auth_manager':
            return module.AuthenticationManager
        elif module_path == 'backend.orchestration.system_orchestrator':
            return module.SystemOrchestrator
        elif module_path == 'backend.diagnostics.system_validator':
            return module.SystemValidator
        return module
    except ImportError as e:
        print(f"Import Error for {module_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

# Attempt imports
TaskQueue = verbose_import('backend.task_queue.task_queue')
ResourceAllocationOptimizer = verbose_import('backend.resource_management.resource_allocator')
MLTaskPredictor = verbose_import('backend.ml.task_predictor')
AuthenticationManager = verbose_import('backend.security.auth_manager')
SystemOrchestrator = verbose_import('backend.orchestration.system_orchestrator')
SystemValidator = verbose_import('backend.diagnostics.system_validator')

class SystemIntegrationTest(unittest.TestCase):
    """
    Comprehensive system integration test suite
    to validate functionality across all components
    """
    
    def setUp(self):
        """
        Initialize system components and validator
        """
        # Initialize components
        self.task_queue = TaskQueue() if TaskQueue else None
        self.resource_allocator = ResourceAllocationOptimizer() if ResourceAllocationOptimizer else None
        self.ml_predictor = MLTaskPredictor() if MLTaskPredictor else None
        self.auth_manager = AuthenticationManager() if AuthenticationManager else None
        self.system_orchestrator = SystemOrchestrator() if SystemOrchestrator else None
        self.system_validator = SystemValidator() if SystemValidator else None
    
    def test_component_imports(self):
        """
        Verify all critical components can be imported
        """
        components = [
            ('Task Queue', TaskQueue),
            ('Resource Allocator', ResourceAllocationOptimizer),
            ('ML Predictor', MLTaskPredictor),
            ('Auth Manager', AuthenticationManager),
            ('System Orchestrator', SystemOrchestrator),
            ('System Validator', SystemValidator)
        ]
        
        for name, component in components:
            with self.subTest(component=name):
                self.assertIsNotNone(
                    component, 
                    f"{name} import failed"
                )
    
    async def async_test_user_registration_flow(self):
        """
        Test complete user registration and authentication flow
        """
        if not self.auth_manager:
            self.skipTest("Authentication Manager not available")
        
        # Register a test user
        registration = await self.auth_manager.register_user(
            username='test_user', 
            password='secure_password', 
            email='test@example.com'
        )
        
        # Validate registration
        self.assertEqual(
            registration['status'], 
            'success', 
            "User registration failed"
        )
        self.assertIn('user_id', registration)
        
        # Authenticate the user
        authentication = await self.auth_manager.authenticate_user(
            username='test_user', 
            password='secure_password'
        )
        
        # Validate authentication
        self.assertEqual(
            authentication['status'], 
            'success', 
            "User authentication failed"
        )
        self.assertIn('access_token', authentication)
    
    def test_user_registration_flow(self):
        """
        Wrapper for async user registration test
        """
        if not self.auth_manager:
            self.skipTest("Authentication Manager not available")
        
        asyncio.run(self.async_test_user_registration_flow())
    
    async def async_test_task_prediction_flow(self):
        """
        Test complete task prediction flow
        """
        if not self.ml_predictor:
            self.skipTest("ML Predictor not available")
        
        # Simulate a computational task
        task_features = {
            'compute_intensity': 75,
            'memory_requirement': 60,
            'network_dependency': 0.5,
            'data_volume': 100,
            'parallelizability': 0.8
        }
        
        # Predict task complexity
        prediction = await self.ml_predictor.predict_task_complexity(task_features)
        
        # Validate prediction
        self.assertEqual(prediction['status'], 'success')
        self.assertIn('complexity_level', prediction)
        self.assertIn('raw_complexity_score', prediction)
        
        # Optional: Add more specific checks based on the new prediction structure
        complexity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        self.assertIn(prediction['complexity_level'], complexity_levels)
        self.assertTrue(0 <= prediction['raw_complexity_score'] <= 3)
    
    def test_task_prediction_flow(self):
        """
        Wrapper for async task prediction test
        """
        if not self.ml_predictor:
            self.skipTest("ML Predictor not available")
        
        asyncio.run(self.async_test_task_prediction_flow())
    
    async def async_test_system_validation(self):
        """
        Run comprehensive system validation
        """
        if not self.system_validator:
            self.skipTest("System Validator not available")
        
        # Run system validator
        validation_report = await self.system_validator.run_comprehensive_validation()
        
        # Validate overall system status
        self.assertIn(
            validation_report['overall_system_status'], 
            ['HEALTHY', 'DEGRADED'], 
            "Unexpected system status"
        )
        
        # Check component validations
        self.assertTrue(
            len(validation_report['component_validations']) > 0,
            "No component validations performed"
        )
    
    def test_system_validation(self):
        """
        Wrapper for async system validation test
        """
        if not self.system_validator:
            self.skipTest("System Validator not available")
        
        asyncio.run(self.async_test_system_validation())

def main():
    """
    Run test suite
    """
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()
