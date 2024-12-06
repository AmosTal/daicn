from pydantic import BaseModel, Field, ConfigDict, validator
from typing import Optional
from datetime import datetime

class ComputeProviderBase(BaseModel):
    wallet_address: str
    compute_power: float = Field(..., gt=0, description="Compute power in FLOPS")
    reputation_score: Optional[float] = 100.0
    is_active: Optional[bool] = True

class ComputeProviderCreate(ComputeProviderBase):
    pass

class ComputeProviderResponse(ComputeProviderBase):
    id: int
    total_tasks_completed: int
    created_at: datetime

    class Config:
        orm_mode = True

class ComputeTaskBase(BaseModel):
    client_address: str
    task_type: str = Field(..., description="Type of AI task")
    input_data_hash: str
    compute_units_required: float = Field(..., gt=0)
    reward_amount: float = Field(..., gt=0)

class ComputeTaskCreate(ComputeTaskBase):
    pass

class ComputeTaskResponse(ComputeTaskBase):
    id: int
    provider_id: Optional[int]
    status: str
    result_hash: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class AIModelBase(BaseModel):
    name: str
    model_type: str
    framework: str
    model_hash: str
    size_mb: float

class AIModelCreate(AIModelBase):
    pass

class AIModelResponse(AIModelBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class TaskAllocationRequest(BaseModel):
    """
    Schema for requesting task allocation in the decentralized network
    """
    task_type: str = Field(..., description="Type of task (TRAINING, INFERENCE, DATA_PROCESSING)")
    required_compute_power: float = Field(..., description="Compute power required for the task")
    user_id: str = Field(..., description="ID of the user requesting the task")
    model_architecture: Optional[str] = Field(None, description="Architecture of the AI model")
    provider_address: Optional[str] = Field(None, description="Optional provider address")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "task_type": "TRAINING",
                "required_compute_power": 50.0,
                "user_id": "user_123",
                "model_architecture": "TRANSFORMER"
            }
        }
    )

class TaskAllocationResponse(BaseModel):
    """
    Schema for task allocation response
    """
    task_id: str
    provider_id: str
    status: str = "ALLOCATED"
    estimated_completion_time: Optional[datetime] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "task_id": "task_456",
                "provider_id": "provider_789",
                "status": "ALLOCATED"
            }
        }
    )

class ComputeProviderSchema(BaseModel):
    """
    Schema for representing a compute provider
    """
    id: str
    name: str
    compute_power: float
    max_concurrent_tasks: int
    reputation_score: float
    is_active: bool
    location: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "provider_123",
                "name": "High Performance Cluster",
                "compute_power": 75.5,
                "max_concurrent_tasks": 15,
                "reputation_score": 0.9,
                "is_active": True,
                "location": "US-WEST"
            }
        }
    )

class ComputeTaskSchema(BaseModel):
    """
    Schema for representing a compute task
    """
    id: str
    user_id: str
    task_type: str
    model_architecture: Optional[str] = None
    required_compute_power: float
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    provider_id: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "task_789",
                "user_id": "user_123",
                "task_type": "TRAINING",
                "model_architecture": "TRANSFORMER",
                "required_compute_power": 50.0,
                "status": "PENDING",
                "created_at": "2024-01-01T12:00:00Z"
            }
        }
    )
