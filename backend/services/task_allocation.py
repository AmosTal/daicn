import random
import logging
from sqlalchemy import func
from backend.models import ComputeProvider, ComputeTask
from backend.services.reputation import ReputationService
from backend.schemas import TaskAllocationRequest, TaskAllocationResponse

class TaskAllocationService:
    def __init__(self, db_session):
        """
        Initialize task allocation service with database session
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.reputation_service = ReputationService(db_session)
        self.logger = logging.getLogger(__name__)
    
    def find_suitable_provider(self, task_request: TaskAllocationRequest) -> TaskAllocationResponse:
        """
        Find a suitable compute provider for the given task
        
        Args:
            task_request: Task allocation request details
        
        Returns:
            TaskAllocationResponse with allocated provider details
        """
        # Query active providers with sufficient compute power
        providers = (
            self.db_session.query(ComputeProvider)
            .filter(
                ComputeProvider.is_active == True,
                ComputeProvider.compute_power >= task_request.required_compute_power
            )
            .order_by(
                ComputeProvider.reputation_score.desc()
            )
            .limit(5)
            .all()
        )
        
        if not providers:
            self.logger.warning("No suitable providers found for task")
            raise ValueError("No suitable providers available")
        
        # Select the top provider
        selected_provider = providers[0]
        
        try:
            # Create a new compute task
            new_task = ComputeTask(
                user_id=task_request.user_id,
                task_type=task_request.task_type,
                model_architecture=task_request.model_architecture,
                required_compute_power=task_request.required_compute_power,
                provider_id=selected_provider.id,
                status='ALLOCATED'
            )
            
            self.db_session.add(new_task)
            self.db_session.commit()
            self.db_session.refresh(new_task)
            
            # Return allocation response
            return TaskAllocationResponse(
                task_id=new_task.id,
                provider_id=selected_provider.id,
                status='ALLOCATED'
            )
        
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Task allocation failed: {str(e)}")
            raise
    
    def _calculate_reward(self, task_request: TaskAllocationRequest) -> float:
        """
        Calculate task reward based on compute requirements and task type
        
        Args:
            task_request: Task allocation request details
        
        Returns:
            Calculated reward amount
        """
        # Base rate for compute power
        BASE_RATE = 0.1
        
        # Task type multipliers
        TASK_TYPE_MULTIPLIERS = {
            'TRAINING': 1.5,
            'INFERENCE': 1.0,
            'DATA_PROCESSING': 0.8
        }
        
        # Calculate task type multiplier
        task_type_multiplier = TASK_TYPE_MULTIPLIERS.get(task_request.task_type, 1.0)
        
        return task_request.required_compute_power * BASE_RATE * task_type_multiplier
    
    def update_task_status(self, task_id: str, status: str):
        """
        Update the status of a compute task
        
        Args:
            task_id: ID of the task
            status: New status of the task
        """
        task = self.db_session.query(ComputeTask).filter(ComputeTask.id == task_id).first()
        if task:
            task.status = status
            self.db_session.commit()
            self.logger.info(f"Task {task_id} status updated to {status}")
        else:
            self.logger.warning(f"Task {task_id} not found")
