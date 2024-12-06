from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
import uvicorn
import asyncio

from . import models, schemas, database
from .services.task_allocation import TaskAllocationService
from .services.reputation_service import ReputationService
from .services.monitoring import NetworkMonitoringService, MonitoringDashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DAICN Compute Marketplace API",
    description="API for decentralized AI computation network",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database initialization
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    # Initialize services
    db = next(get_db())
    
    # Initialize monitoring service
    monitoring_service = NetworkMonitoringService(db)
    
    # Start background monitoring tasks
    asyncio.create_task(monitoring_service.start_monitoring())
    asyncio.create_task(
        MonitoringDashboard.start_dashboard_monitoring(monitoring_service)
    )

    database.init_db()
    logger.info("Database initialized")

# Providers Endpoints
@app.post("/providers/", response_model=schemas.ComputeProviderResponse)
def register_provider(
    provider: schemas.ComputeProviderCreate, 
    db: Session = Depends(get_db)
):
    """
    Register a new compute provider in the network
    """
    try:
        db_provider = models.ComputeProvider(
            wallet_address=provider.wallet_address,
            compute_power=provider.compute_power,
            reputation_score=provider.reputation_score or 100.0,
            is_active=provider.is_active or True
        )
        db.add(db_provider)
        db.commit()
        db.refresh(db_provider)
        return db_provider
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tasks/allocate", response_model=schemas.TaskAllocationResponse)
def allocate_task(
    task_request: schemas.TaskAllocationRequest, 
    db: Session = Depends(get_db)
):
    """
    Allocate a compute task to a suitable provider
    """
    try:
        task_service = TaskAllocationService(db)
        return task_service.allocate_task(task_request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/{task_id}/complete")
def complete_task(
    task_id: int, 
    result_hash: str, 
    db: Session = Depends(get_db)
):
    """
    Mark a task as completed and update provider reputation
    """
    try:
        task = db.query(models.ComputeTask).get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Update task status
        task.status = 'completed'
        task.result_hash = result_hash
        
        # Update provider reputation
        reputation_service = ReputationService(db)
        reputation_service.update_provider_reputation(task.provider_id, task_success=True)
        
        db.commit()
        return {"status": "success", "message": "Task completed"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers/{provider_id}/performance")
def get_provider_performance(
    provider_id: int, 
    db: Session = Depends(get_db)
):
    """
    Get performance metrics for a specific provider
    """
    try:
        reputation_service = ReputationService(db)
        performance = reputation_service.calculate_provider_performance(provider_id)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network/metrics")
def get_network_metrics(db: Session = Depends(get_db)):
    monitoring_service = NetworkMonitoringService(db)
    return monitoring_service.get_current_network_metrics()

@app.get("/network/health")
def get_network_health(db: Session = Depends(get_db)):
    monitoring_service = NetworkMonitoringService(db)
    return monitoring_service.generate_network_health_report()

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
