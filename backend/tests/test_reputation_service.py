import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from ..models import Base, ComputeProvider, ComputeTask
from ..services.reputation_service import ReputationService

class TestReputationService:
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
    def reputation_service(self, db_session):
        return ReputationService(db_session)

    def test_provider_reputation_update(self, db_session, reputation_service):
        """
        Test reputation update for successful and failed tasks
        """
        # Create a test provider
        provider = ComputeProvider(
            wallet_address='test_provider@example.com',
            compute_power=1000,
            reputation_score=100.0,
            is_active=True
        )
        db_session.add(provider)
        db_session.commit()

        # Test successful task
        new_reputation = reputation_service.update_provider_reputation(provider.id, task_success=True)
        assert new_reputation > 100.0
        assert new_reputation <= 100.0  # Boost is limited

        # Test failed task
        new_reputation = reputation_service.update_provider_reputation(provider.id, task_success=False)
        assert new_reputation < 100.0
        assert new_reputation >= 0.0  # Penalty has a lower bound

    def test_provider_performance_calculation(self, db_session, reputation_service):
        """
        Test performance calculation over a time window
        """
        # Create a test provider
        provider = ComputeProvider(
            wallet_address='performance_provider@example.com',
            compute_power=1000,
            reputation_score=100.0,
            is_active=True
        )
        db_session.add(provider)
        db_session.commit()

        # Create sample tasks
        tasks = [
            ComputeTask(
                provider_id=provider.id,
                client_address='client@example.com',
                task_type='training',
                status='completed' if i % 2 == 0 else 'failed',
                created_at=datetime.utcnow() - timedelta(days=i)
            ) for i in range(10)
        ]
        db_session.add_all(tasks)
        db_session.commit()

        # Calculate performance
        performance = reputation_service.calculate_provider_performance(provider.id)
        
        assert 'total_tasks' in performance
        assert 'successful_tasks' in performance
        assert 'performance_percentage' in performance
        assert performance['total_tasks'] == 10
        assert performance['successful_tasks'] == 5
        assert performance['performance_percentage'] == 50.0

    def test_reputation_decay(self, db_session, reputation_service):
        """
        Test reputation decay for inactive providers
        """
        # Create inactive providers with different initial reputations
        providers = [
            ComputeProvider(
                wallet_address=f'inactive_provider_{i}@example.com',
                compute_power=1000,
                reputation_score=float(i * 20),
                is_active=False
            ) for i in range(1, 6)
        ]
        db_session.add_all(providers)
        db_session.commit()

        # Apply reputation decay
        reputation_service.apply_reputation_decay()
        
        # Verify reputation has decayed
        for provider in db_session.query(ComputeProvider).filter(ComputeProvider.is_active == False):
            assert provider.reputation_score < (provider.id * 20)
            assert provider.reputation_score >= 0

    def test_provider_deactivation(self, db_session, reputation_service):
        """
        Test automatic provider deactivation for very low reputation
        """
        # Create a provider with low initial reputation
        provider = ComputeProvider(
            wallet_address='low_rep_provider@example.com',
            compute_power=1000,
            reputation_score=15.0,
            is_active=True
        )
        db_session.add(provider)
        db_session.commit()

        # Simulate multiple failed tasks to reduce reputation
        for _ in range(5):
            reputation_service.update_provider_reputation(provider.id, task_success=False)

        db_session.refresh(provider)
        
        # Provider should be deactivated
        assert provider.is_active == False
        assert provider.reputation_score < 20.0
