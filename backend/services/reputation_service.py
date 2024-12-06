from sqlalchemy.orm import Session
from ..models import ComputeProvider, ComputeTask
import logging
from datetime import datetime, timedelta

class ReputationService:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def update_provider_reputation(self, provider_id: int, task_success: bool):
        """
        Update provider's reputation based on task performance
        """
        provider = self.db.query(ComputeProvider).get(provider_id)
        if not provider:
            self.logger.warning(f"Provider {provider_id} not found")
            return

        # Reputation adjustment parameters
        SUCCESS_BOOST = 2.0
        FAILURE_PENALTY = 5.0
        MAX_REPUTATION = 100.0
        MIN_REPUTATION = 0.0

        # Calculate reputation change
        if task_success:
            # Boost reputation for successful tasks
            reputation_change = SUCCESS_BOOST
        else:
            # Penalize for failed tasks
            reputation_change = -FAILURE_PENALTY

        # Update reputation with bounds
        new_reputation = max(
            MIN_REPUTATION, 
            min(MAX_REPUTATION, provider.reputation_score + reputation_change)
        )

        provider.reputation_score = new_reputation
        provider.total_tasks_completed += 1 if task_success else 0

        # Potentially deactivate very low-reputation providers
        if new_reputation < 20.0:
            provider.is_active = False
            self.logger.warning(f"Provider {provider_id} deactivated due to low reputation")

        self.db.commit()
        return new_reputation

    def calculate_provider_performance(self, provider_id: int, time_window: int = 30):
        """
        Calculate provider performance over a given time window
        """
        cutoff_date = datetime.utcnow() - timedelta(days=time_window)
        
        # Get total and successful tasks
        total_tasks = self.db.query(ComputeTask).filter(
            ComputeTask.provider_id == provider_id,
            ComputeTask.created_at >= cutoff_date
        ).count()

        successful_tasks = self.db.query(ComputeTask).filter(
            ComputeTask.provider_id == provider_id,
            ComputeTask.status == 'completed',
            ComputeTask.created_at >= cutoff_date
        ).count()

        # Calculate performance percentage
        performance_percentage = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0

        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'performance_percentage': performance_percentage
        }

    def apply_reputation_decay(self):
        """
        Apply gradual reputation decay for inactive providers
        """
        providers = self.db.query(ComputeProvider).filter(
            ComputeProvider.is_active == False,
            ComputeProvider.reputation_score > 0
        ).all()

        for provider in providers:
            # Slow decay for inactive providers
            provider.reputation_score = max(0, provider.reputation_score - 0.1)

        self.db.commit()
