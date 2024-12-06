import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..models import Base, ComputeProvider, ComputeTask
from ..services.task_allocation import TaskAllocationService
from ..schemas import TaskAllocationRequest

class TestTaskAllocationService:
    @pytest.fixture(scope="function")
    def db_session(self):
        # Use an in-memory SQLite database for testing
        engine = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False})
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Create a new database session
        session = TestingSessionLocal()
        
        try:
            yield session
        finally:
            session.close()

    @pytest.fixture
    def task_allocation_service(self, db_session):
        return TaskAllocationService(db_session)

    def test_provider_creation(self, db_session, task_allocation_service):
        """
        Test creating and finding suitable providers
        """
        # Create multiple providers with different capabilities
        providers = [
            ComputeProvider(
                wallet_address=f'provider{i}@example.com', 
                compute_power=1000 * (i+1), 
                reputation_score=80 + i*2,
                is_active=True
            ) for i in range(3)
        ]
        
        db_session.add_all(providers)
        db_session.commit()

        # Create a task allocation request
        task_request = TaskAllocationRequest(
            provider_address='client@example.com',
            required_compute_power=2000,
            task_type='training'
        )

        # Find suitable provider
        selected_provider = task_allocation_service.find_suitable_provider(task_request)
        
        assert selected_provider is not None
        assert selected_provider.compute_power >= task_request.required_compute_power
        assert selected_provider.is_active

    def test_task_allocation(self, db_session, task_allocation_service):
        """
        Test complete task allocation process
        """
        # Create providers
        providers = [
            ComputeProvider(
                wallet_address=f'provider{i}@example.com', 
                compute_power=1000 * (i+1), 
                reputation_score=80 + i*2,
                is_active=True
            ) for i in range(3)
        ]
        
        db_session.add_all(providers)
        db_session.commit()

        # Create task allocation request
        task_request = TaskAllocationRequest(
            provider_address='client@example.com',
            required_compute_power=2000,
            task_type='training'
        )

        # Allocate task
        task_allocation = task_allocation_service.allocate_task(task_request)
        
        assert task_allocation is not None
        assert task_allocation.task_id is not None
        assert task_allocation.compute_units > 0
        assert task_allocation.reward > 0

    def test_reward_calculation(self, task_allocation_service):
        """
        Test reward calculation for different task types
        """
        test_cases = [
            {'task_type': 'training', 'compute_power': 1000, 'expected_multiplier': 1.5},
            {'task_type': 'inference', 'compute_power': 500, 'expected_multiplier': 1.0},
            {'task_type': 'complex_inference', 'compute_power': 2000, 'expected_multiplier': 2.0}
        ]

        for case in test_cases:
            # Use reflection to test private method
            reward = task_allocation_service._calculate_reward(
                TaskAllocationRequest(
                    provider_address='test@example.com',
                    required_compute_power=case['compute_power'],
                    task_type=case['task_type']
                )
            )

            # Base rate is 0.1, so expected reward = compute_power * 0.1 * task_type_multiplier
            expected_reward = case['compute_power'] * 0.1 * case['expected_multiplier']
            
            assert abs(reward - expected_reward) < 0.01

    def test_no_suitable_provider(self, db_session, task_allocation_service):
        """
        Test scenario where no suitable provider exists
        """
        # Create providers that are too weak or inactive
        providers = [
            ComputeProvider(
                wallet_address=f'provider{i}@example.com', 
                compute_power=500, 
                reputation_score=50,
                is_active=False if i % 2 == 0 else True
            ) for i in range(3)
        ]
        
        db_session.add_all(providers)
        db_session.commit()

        # Create task allocation request
        task_request = TaskAllocationRequest(
            provider_address='client@example.com',
            required_compute_power=2000,
            task_type='training'
        )

        # Expect a ValueError when no suitable provider is found
        with pytest.raises(ValueError, match="No suitable providers found for the task"):
            task_allocation_service.allocate_task(task_request)
