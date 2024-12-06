import os
import sys
import logging
import random
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import SessionLocal, init_db
from backend.models import ComputeProvider, ComputeTask
from backend.services.task_allocation import TaskAllocationService
from backend.services.monitoring import NetworkMonitoringService
from backend.services.reputation import ReputationService
from backend.schemas import TaskAllocationRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('network_simulation.log')
    ]
)

def create_compute_providers(db_session, num_providers=10):
    """
    Create simulated compute providers
    """
    for i in range(num_providers):
        provider = ComputeProvider(
            name=f"Provider_{i+1}",
            compute_power=random.uniform(10, 100),
            max_concurrent_tasks=random.randint(5, 20),
            is_active=random.choice([True, True, True, False]),  # More likely to be active
            location=random.choice(['US', 'EU', 'ASIA', 'OCEANIA']),
            reputation_score=random.uniform(0.5, 1.0)
        )
        db_session.add(provider)
    
    db_session.commit()
    logging.info(f"Created {num_providers} compute providers")

def simulate_task_allocation(db_session):
    """
    Simulate task allocation across network
    """
    task_allocation_service = TaskAllocationService(db_session)
    reputation_service = ReputationService(db_session)
    
    # Get all active providers
    providers = db_session.query(ComputeProvider).filter(ComputeProvider.is_active == True).all()
    
    if not providers:
        logging.warning("No active providers to simulate tasks")
        return
    
    # Simulate multiple task types
    task_types = ['TRAINING', 'INFERENCE', 'DATA_PROCESSING']
    
    for _ in range(20):  # Simulate 20 tasks
        task_request = TaskAllocationRequest(
            task_type=random.choice(task_types),
            required_compute_power=random.uniform(5, 50),
            user_id=f"user_{random.randint(1, 100)}",
            model_architecture=random.choice(['TRANSFORMER', 'CNN', 'RNN', 'GAN'])
        )
        
        try:
            # Allocate task to a provider
            allocation_result = task_allocation_service.find_suitable_provider(task_request)
            
            # Simulate task success probabilistically
            task_success = random.random() > 0.2  # 80% success rate
            
            # Update provider reputation
            reputation_service.update_provider_reputation(
                provider_id=allocation_result.provider_id, 
                task_success=task_success
            )
            
            logging.info(f"Task allocated: {allocation_result}, Success: {task_success}")
        
        except Exception as e:
            logging.error(f"Task allocation failed: {e}")

def main():
    """
    Main simulation entry point
    """
    # Initialize database
    init_db()
    
    # Create database session
    db_session = SessionLocal()
    
    try:
        # Create compute providers
        create_compute_providers(db_session)
        
        # Simulate task allocation
        simulate_task_allocation(db_session)
        
        # Run network monitoring
        monitoring_service = NetworkMonitoringService(db_session)
        network_metrics = monitoring_service.get_network_metrics()
        
        logging.info("Network Metrics:")
        for key, value in network_metrics.items():
            logging.info(f"{key}: {value}")
    
    except Exception as e:
        logging.error(f"Simulation error: {e}")
    
    finally:
        db_session.close()

if __name__ == "__main__":
    main()
