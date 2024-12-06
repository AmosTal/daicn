from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base
import uuid

def generate_uuid():
    """Generate a unique identifier"""
    return str(uuid.uuid4())

class ComputeProvider(Base):
    """
    Represents a compute provider in the decentralized network
    """
    __tablename__ = 'compute_providers'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    wallet_address = Column(String, unique=True, nullable=True)
    
    # Provider details
    name = Column(String, nullable=False)
    compute_power = Column(Float, nullable=False)
    max_concurrent_tasks = Column(Integer, default=10)
    reputation_score = Column(Float, default=0.5)
    is_active = Column(Boolean, default=True)
    location = Column(String, nullable=True)
    
    # Relationships
    tasks = relationship('ComputeTask', back_populates='provider')
    
    def __repr__(self):
        return f"<ComputeProvider(id={self.id}, name={self.name}, compute_power={self.compute_power})>"

class ComputeTask(Base):
    """
    Represents a compute task in the decentralized network
    """
    __tablename__ = 'compute_tasks'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Task details
    user_id = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # e.g., TRAINING, INFERENCE
    model_architecture = Column(String, nullable=True)
    required_compute_power = Column(Float, nullable=False)
    status = Column(String, default='PENDING')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Provider relationship
    provider_id = Column(String, ForeignKey('compute_providers.id'), nullable=True)
    provider = relationship('ComputeProvider', back_populates='tasks')
    
    def __repr__(self):
        return f"<ComputeTask(id={self.id}, task_type={self.task_type}, status={self.status})>"
