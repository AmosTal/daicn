import pytest
import numpy as np
import torch
import tensorflow as tf

from ..services.task_processing import TaskProcessingService

class TestTaskProcessingService:
    @pytest.fixture
    def task_processing_service(self):
        return TaskProcessingService()

    @pytest.mark.asyncio
    async def test_pytorch_training_task(self, task_processing_service):
        """
        Test PyTorch model training task
        """
        task_data = {
            'framework': 'pytorch',
            'task_type': 'training',
            'model_config': {
                'type': 'linear',
                'input_size': 10,
                'output_size': 1
            },
            'input_data': np.random.rand(100, 10).tolist(),
            'hyperparameters': {
                'learning_rate': 0.001,
                'epochs': 5
            }
        }

        result = await task_processing_service.process_ai_task(task_data)
        
        assert result['status'] == 'completed'
        assert 'result' in result
        assert 'result_hash' in result

    @pytest.mark.asyncio
    async def test_tensorflow_inference_task(self, task_processing_service):
        """
        Test TensorFlow model inference task
        """
        task_data = {
            'framework': 'tensorflow',
            'task_type': 'inference',
            'model_config': {
                'type': 'sequential',
                'input_size': 10,
                'output_size': 1
            },
            'input_data': np.random.rand(50, 10).tolist()
        }

        result = await task_processing_service.process_ai_task(task_data)
        
        assert result['status'] == 'completed'
        assert 'result' in result
        assert 'result_hash' in result

    def test_result_hash_generation(self, task_processing_service):
        """
        Test result hash generation
        """
        sample_result = {'data': [1, 2, 3], 'model_weights': [0.1, 0.2, 0.3]}
        result_hash = task_processing_service._generate_result_hash(sample_result)
        
        assert isinstance(result_hash, str)
        assert len(result_hash) == 64  # SHA-256 hash length

    @pytest.mark.parametrize("invalid_task", [
        # Missing framework
        {'task_type': 'training', 'input_data': [[1,2,3]]},
        # Unsupported framework
        {'framework': 'unsupported', 'task_type': 'training', 'input_data': [[1,2,3]]},
        # Empty input data
        {'framework': 'pytorch', 'task_type': 'training', 'input_data': []},
    ])
    @pytest.mark.asyncio
    async def test_invalid_task_handling(self, task_processing_service, invalid_task):
        """
        Test handling of invalid task configurations
        """
        with pytest.raises((ValueError, KeyError)):
            await task_processing_service.process_ai_task(invalid_task)

    def test_model_creation(self, task_processing_service):
        """
        Test model creation for different frameworks
        """
        # PyTorch model creation
        pytorch_model = task_processing_service._create_pytorch_model({
            'type': 'linear',
            'input_size': 10,
            'output_size': 1
        })
        assert isinstance(pytorch_model, torch.nn.Module)

        # TensorFlow model creation
        tensorflow_model = task_processing_service._create_tensorflow_model({
            'type': 'sequential',
            'input_size': 10,
            'output_size': 1
        })
        assert isinstance(tensorflow_model, tf.keras.Model)

    @pytest.mark.asyncio
    async def test_complex_ai_task(self, task_processing_service):
        """
        Test a more complex AI task with multiple configurations
        """
        complex_task = {
            'framework': 'pytorch',
            'task_type': 'training',
            'model_config': {
                'type': 'linear',
                'input_size': 20,
                'output_size': 5
            },
            'input_data': np.random.rand(200, 20).tolist(),
            'hyperparameters': {
                'learning_rate': 0.005,
                'epochs': 10,
                'batch_size': 32
            }
        }

        result = await task_processing_service.process_ai_task(complex_task)
        
        assert result['status'] == 'completed'
        assert 'result' in result
        assert 'result_hash' in result
