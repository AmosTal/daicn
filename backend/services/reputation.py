import logging
from backend.models import ComputeProvider

class ReputationService:
    def __init__(self, db_session):
        """
        Initialize reputation service with database session
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
    
    def get_provider_reputation(self, provider_id: str) -> float:
        """
        Retrieve the reputation score for a compute provider
        
        Args:
            provider_id: ID of the compute provider
        
        Returns:
            float: Reputation score between 0 and 1
        """
        provider = self.db_session.query(ComputeProvider).filter(ComputeProvider.id == provider_id).first()
        
        if not provider:
            self.logger.warning(f"Provider {provider_id} not found")
            return 0.5  # Default neutral reputation
        
        return provider.reputation_score
    
    def update_provider_reputation(self, provider_id: str, task_success: bool):
        """
        Update provider's reputation based on task performance
        
        Args:
            provider_id: ID of the compute provider
            task_success: Whether the task was successfully completed
        """
        provider = self.db_session.query(ComputeProvider).filter(ComputeProvider.id == provider_id).first()
        
        if not provider:
            self.logger.warning(f"Provider {provider_id} not found")
            return
        
        # Reputation adjustment parameters
        SUCCESS_WEIGHT = 0.1
        FAILURE_WEIGHT = 0.2
        
        if task_success:
            # Increase reputation for successful tasks
            provider.reputation_score = min(
                1.0, 
                provider.reputation_score + SUCCESS_WEIGHT
            )
        else:
            # Decrease reputation for failed tasks
            provider.reputation_score = max(
                0.0, 
                provider.reputation_score - FAILURE_WEIGHT
            )
        
        try:
            self.db_session.commit()
            self.logger.info(f"Provider {provider_id} reputation updated to {provider.reputation_score}")
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Failed to update reputation for provider {provider_id}: {e}")
    
    def penalize_inactive_providers(self):
        """
        Penalize providers who have been inactive or have low performance
        """
        inactive_providers = self.db_session.query(ComputeProvider).filter(
            ComputeProvider.is_active == False
        ).all()
        
        for provider in inactive_providers:
            # Significant reputation penalty for inactive providers
            provider.reputation_score = max(
                0.0, 
                provider.reputation_score - 0.3
            )
        
        try:
            self.db_session.commit()
            self.logger.info(f"Penalized {len(inactive_providers)} inactive providers")
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Failed to penalize inactive providers: {e}")
