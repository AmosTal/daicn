import os
import sys
import logging
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import threading
import time
import random
import datetime

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project-specific modules
try:
    from backend.database import DATABASE_URL, Base
    from backend.models import ComputeProvider, ComputeTask
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, async_mode='threading')

# Database setup
try:
    engine = create_engine(DATABASE_URL, echo=True)
    SessionLocal = sessionmaker(bind=engine)
except Exception as e:
    logger.error(f"Database engine creation failed: {e}")
    raise

# Global state for synchronized metrics
class NetworkState:
    def __init__(self):
        self.total_providers = 0
        self.active_providers = 0
        self.total_tasks = 0
        self.average_provider_reputation = 0.0
        self.health_score = 0.0
        self.status = 'UNKNOWN'
        self.active_provider_ratio = 0.0
        self.timestamp = None

    def generate_metrics(self):
        """Generate synchronized network metrics."""
        # Use current second as seed for more frequent updates
        random.seed(int(time.time()))
        
        # Base metrics with some controlled randomness
        base_total_providers = 200  # Stable base
        base_active_ratio = 0.4     # Stable base ratio
        
        # Generate metrics with controlled variation
        self.total_providers = max(50, min(250, 
            base_total_providers + random.randint(-20, 20)
        ))
        self.active_providers = max(20, min(self.total_providers, 
            int(self.total_providers * (base_active_ratio + random.uniform(-0.1, 0.1)))
        ))
        
        # Tasks generation with relationship to providers
        self.total_tasks = max(100, min(600, 
            int(self.active_providers * random.uniform(2, 5))
        ))
        
        # Reputation with controlled variance
        self.average_provider_reputation = round(
            max(3.0, min(5.0, 
                4.0 + random.uniform(-0.5, 0.5)
            )), 2
        )
        
        # Health calculation with more nuanced approach
        self.active_provider_ratio = round((self.active_providers / self.total_providers) * 100, 2)
        self.health_score = round(
            (self.active_provider_ratio * 0.5) +  # Provider activity
            (self.average_provider_reputation * 10) +  # Reputation impact
            (self.total_tasks / 10) +  # Task volume influence
            random.uniform(-5, 5),  # Some controlled randomness
            2
        )
        
        # Constrain health score
        self.health_score = max(0, min(100, self.health_score))
        
        # Determine status
        status_levels = [
            (0, 20, 'CRITICAL'),
            (20, 40, 'POOR'),
            (40, 60, 'MODERATE'),
            (60, 80, 'GOOD'),
            (80, 100, 'EXCELLENT')
        ]
        
        for min_score, max_score, status in status_levels:
            if min_score <= self.health_score < max_score:
                self.status = status
                break
        
        self.timestamp = datetime.datetime.now().isoformat()
        
        return {
            'total_providers': self.total_providers,
            'active_providers': self.active_providers,
            'total_tasks': self.total_tasks,
            'average_provider_reputation': self.average_provider_reputation,
            'health_score': self.health_score,
            'status': self.status,
            'active_provider_ratio': self.active_provider_ratio,
            'timestamp': self.timestamp
        }

# Create a global network state
network_state = NetworkState()

def background_metrics_generator():
    """Background thread to emit metrics periodically."""
    global network_state
    logger.info("Starting background metrics generator")
    
    while True:
        try:
            # Generate metrics every 1 second
            metrics = network_state.generate_metrics()
            
            # Broadcast to all connected clients
            socketio.emit('network_metrics_update', metrics, namespace='/dashboard')
            socketio.emit('network_health_update', metrics, namespace='/dashboard')
            
            logger.info("Emitted metrics and health updates")
            
            # Sleep for 1 second
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in background metrics generator: {e}")
            time.sleep(1)  # Wait before retrying

@app.route('/')
def dashboard():
    """
    Render the main dashboard page
    """
    return render_template('index.html')

@app.route('/api/network-metrics')
def get_network_metrics():
    """
    API endpoint to fetch network metrics
    """
    try:
        # Create a new database session
        db_session = SessionLocal()
        
        # Collect metrics
        total_providers = db_session.query(ComputeProvider).count()
        active_providers = db_session.query(ComputeProvider).filter(ComputeProvider.is_active == True).count()
        total_tasks = db_session.query(ComputeTask).count()
        
        # Tasks by status
        tasks_by_status = (
            db_session.query(
                ComputeTask.status, 
                func.count(ComputeTask.id).label('count')
            )
            .group_by(ComputeTask.status)
            .all()
        )
        
        # Average provider reputation
        avg_reputation = (
            db_session.query(func.avg(ComputeProvider.reputation_score))
            .scalar() or 0
        )
        
        # Close the session
        db_session.close()
        
        # Prepare metrics dictionary
        metrics = {
            'total_providers': total_providers,
            'active_providers': active_providers,
            'total_tasks': total_tasks,
            'average_provider_reputation': round(avg_reputation, 2),
            'tasks_by_status': dict(tasks_by_status)
        }
        
        logger.info(f"Network metrics collected: {metrics}")
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Error fetching network metrics: {e}")
        return jsonify({
            "error": "Could not fetch network metrics",
            "details": str(e)
        }), 500

@app.route('/api/network-health')
def get_network_health():
    """
    API endpoint to fetch network health with comprehensive validation
    """
    try:
        # Create a new database session
        db_session = SessionLocal()
        
        # Validate database connection
        if not db_session:
            logger.error("Failed to establish database session")
            return jsonify({
                "error": "Database connection failed",
                "health_score": 0,
                "status": "CRITICAL"
            }), 500
        
        try:
            # Collect health metrics with error handling
            total_providers = db_session.query(ComputeProvider).count()
            active_providers = db_session.query(ComputeProvider).filter(ComputeProvider.is_active == True).count()
            total_tasks = db_session.query(ComputeTask).count()
            tasks_completed = db_session.query(ComputeTask).filter(ComputeTask.status == 'COMPLETED').count()
            tasks_failed = db_session.query(ComputeTask).filter(ComputeTask.status == 'FAILED').count()
            
            # Validate metrics
            if total_providers is None or active_providers is None:
                logger.warning("Unable to retrieve provider metrics")
                return jsonify({
                    "error": "Provider metrics unavailable",
                    "health_score": 0,
                    "status": "CRITICAL"
                }), 500
            
            # Calculate health score with comprehensive checks
            health_score = 100
            
            # Provider health
            try:
                active_provider_ratio = active_providers / total_providers if total_providers > 0 else 0
                if active_provider_ratio < 0.5:
                    health_score -= 20
                    logger.warning(f"Low active provider ratio: {active_provider_ratio}")
            except ZeroDivisionError:
                logger.error("Zero total providers detected")
                health_score = 0
            
            # Task completion health
            try:
                if total_tasks > 0:
                    completion_ratio = tasks_completed / total_tasks
                    failure_ratio = tasks_failed / total_tasks
                    
                    if completion_ratio < 0.7:
                        health_score -= 15
                        logger.warning(f"Low task completion ratio: {completion_ratio}")
                    
                    if failure_ratio > 0.2:
                        health_score -= 25
                        logger.warning(f"High task failure ratio: {failure_ratio}")
            except ZeroDivisionError:
                logger.error("Zero total tasks detected")
            
            # Ensure health score is within valid range
            health_score = max(0, min(health_score, 100))
            
            # Determine health status with more granular categorization
            if health_score >= 90:
                status = 'EXCELLENT'
            elif health_score >= 75:
                status = 'VERY_GOOD'
            elif health_score >= 60:
                status = 'GOOD'
            elif health_score >= 40:
                status = 'MODERATE'
            elif health_score >= 20:
                status = 'POOR'
            else:
                status = 'CRITICAL'
            
            # Prepare comprehensive health report
            health_report = {
                'health_score': round(health_score, 2),
                'status': status,
                'total_providers': total_providers,
                'active_providers': active_providers,
                'active_provider_ratio': round(active_provider_ratio * 100, 2),
                'total_tasks': total_tasks,
                'tasks_completed': tasks_completed,
                'tasks_failed': tasks_failed,
                'completion_ratio': round((tasks_completed / total_tasks * 100) if total_tasks > 0 else 0, 2),
                'failure_ratio': round((tasks_failed / total_tasks * 100) if total_tasks > 0 else 0, 2)
            }
            
            logger.info(f"Network health assessed: {health_report}")
            return jsonify(health_report)
        
        except SQLAlchemyError as db_error:
            logger.error(f"Database query error: {db_error}")
            return jsonify({
                "error": "Database query failed",
                "health_score": 0,
                "status": "CRITICAL"
            }), 500
        
        finally:
            # Ensure session is always closed
            db_session.close()
    
    except Exception as e:
        logger.error(f"Unexpected error in network health assessment: {e}")
        return jsonify({
            "error": "Unexpected error occurred",
            "health_score": 0,
            "status": "CRITICAL"
        }), 500

@socketio.on('connect', namespace='/dashboard')
def handle_connect():
    logger.info('Client connected to dashboard namespace')
    # Immediately send current metrics on connection
    metrics = network_state.generate_metrics()
    emit('network_metrics_update', metrics)
    emit('network_health_update', metrics)

@socketio.on('disconnect', namespace='/dashboard')
def handle_disconnect():
    logger.info('Client disconnected from dashboard namespace')

if __name__ == '__main__':
    # Start background metrics generator thread
    metrics_thread = threading.Thread(target=background_metrics_generator)
    metrics_thread.daemon = True
    metrics_thread.start()
    
    try:
        # Ensure database tables are created
        Base.metadata.create_all(bind=engine)
        
        # Run the Flask-SocketIO app
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
