import asyncio
import time
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from backend.models import ComputeProvider, ComputeTask

class NetworkMonitoringService:
    """
    Service for monitoring network metrics and health
    """
    def __init__(self, db_session: Session):
        """
        Initialize monitoring service with database session
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self._network_metrics = {
            'total_compute_providers': 0,
            'active_providers': 0,
            'total_tasks_processed': 0,
            'tasks_in_progress': 0,
            'network_utilization': 0.0,
            'average_task_completion_time': 0.0,
            'total_compute_power': 0,
            'avg_provider_reputation': 0.0
        }
        self._historical_metrics = []

    async def start_monitoring(self, update_interval: int = 60):
        """
        Start continuous monitoring of network metrics
        """
        while True:
            try:
                self._update_network_metrics()
                self._record_historical_metrics()
                await asyncio.sleep(update_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(update_interval)

    def _update_network_metrics(self):
        """
        Collect and update current network metrics
        """
        # Compute Providers Metrics
        providers = self.db_session.query(ComputeProvider).all()
        self._network_metrics['total_compute_providers'] = len(providers)
        self._network_metrics['active_providers'] = sum(1 for p in providers if p.is_active)
        self._network_metrics['total_compute_power'] = sum(p.compute_power for p in providers)
        self._network_metrics['avg_provider_reputation'] = sum(p.reputation_score for p in providers) / len(providers) if providers else 0

        # Tasks Metrics
        tasks = self.db_session.query(ComputeTask).all()
        self._network_metrics['total_tasks_processed'] = len(tasks)
        self._network_metrics['tasks_in_progress'] = sum(1 for t in tasks if t.status == 'in_progress')

        # Calculate network utilization and average task completion time
        completed_tasks = [t for t in tasks if t.status == 'completed']
        if completed_tasks:
            completion_times = [(t.completed_at - t.created_at).total_seconds() for t in completed_tasks]
            self._network_metrics['average_task_completion_time'] = sum(completion_times) / len(completion_times)

        # Estimate network utilization (simplified)
        if providers:
            active_compute_power = sum(p.compute_power for p in providers if p.is_active)
            total_compute_power = sum(p.compute_power for p in providers)
            self._network_metrics['network_utilization'] = (active_compute_power / total_compute_power) * 100

    def _record_historical_metrics(self):
        """
        Store historical metrics for trend analysis
        """
        current_metrics = self._network_metrics.copy()
        current_metrics['timestamp'] = datetime.utcnow()
        
        # Keep only last 24 hours of metrics
        self._historical_metrics.append(current_metrics)
        self._historical_metrics = [
            m for m in self._historical_metrics 
            if m['timestamp'] > datetime.utcnow() - timedelta(hours=24)
        ]

    def get_current_network_metrics(self) -> Dict[str, Any]:
        """
        Retrieve current network metrics
        """
        return self._network_metrics.copy()

    def get_historical_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve historical metrics for the last specified hours
        """
        return [
            m for m in self._historical_metrics 
            if m['timestamp'] > datetime.utcnow() - timedelta(hours=hours)
        ]

    def get_network_metrics(self):
        """
        Collect comprehensive network metrics
        
        Returns:
            Dictionary of network performance metrics
        """
        try:
            # Total number of providers
            total_providers = self.db_session.query(ComputeProvider).count()
            
            # Active providers
            active_providers = self.db_session.query(ComputeProvider).filter(ComputeProvider.is_active == True).count()
            
            # Total tasks
            total_tasks = self.db_session.query(ComputeTask).count()
            
            # Tasks by status
            tasks_by_status = (
                self.db_session.query(
                    ComputeTask.status, 
                    func.count(ComputeTask.id).label('count')
                )
                .group_by(ComputeTask.status)
                .all()
            )
            
            # Average provider reputation
            avg_reputation = (
                self.db_session.query(func.avg(ComputeProvider.reputation_score))
                .scalar() or 0
            )
            
            # Metrics dictionary
            metrics = {
                'total_providers': total_providers,
                'active_providers': active_providers,
                'total_tasks': total_tasks,
                'average_provider_reputation': round(avg_reputation, 2),
                'tasks_by_status': {status: count for status, count in tasks_by_status}
            }
            
            self.logger.info("Network metrics collected successfully")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error collecting network metrics: {str(e)}")
            return {}

    def assess_network_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive network health assessment
        
        Returns:
            Dictionary of network health indicators
        """
        try:
            # Provider health metrics
            provider_health = {
                'total_providers': self.db_session.query(ComputeProvider).count(),
                'active_providers_percentage': (
                    self.db_session.query(ComputeProvider)
                    .filter(ComputeProvider.is_active == True)
                    .count() / max(1, self.db_session.query(ComputeProvider).count()) * 100
                ),
                'avg_provider_reputation': self.db_session.query(
                    func.avg(ComputeProvider.reputation_score)
                ).scalar() or 0
            }
            
            # Task processing health
            task_health = {
                'total_tasks': self.db_session.query(ComputeTask).count(),
                'completed_tasks_percentage': (
                    self.db_session.query(ComputeTask)
                    .filter(ComputeTask.status == 'COMPLETED')
                    .count() / max(1, self.db_session.query(ComputeTask).count()) * 100
                )
            }
            
            # Combine health indicators
            network_health = {
                **provider_health,
                **task_health,
                'overall_health_score': (
                    provider_health['active_providers_percentage'] * 0.4 +
                    provider_health['avg_provider_reputation'] * 0.3 +
                    task_health['completed_tasks_percentage'] * 0.3
                )
            }
            
            self.logger.info("Network health assessment completed")
            return network_health
        
        except Exception as e:
            self.logger.error(f"Failed to assess network health: {e}")
            return {}

    def generate_network_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive network health report
        """
        current_metrics = self.get_current_network_metrics()
        network_health = self.assess_network_health()
        
        health_report = {
            'overall_health': 'GOOD',
            'metrics': current_metrics,
            'health_indicators': network_health,
            'recommendations': []
        }

        # Simple health assessment logic
        if current_metrics['network_utilization'] > 90:
            health_report['overall_health'] = 'CRITICAL'
            health_report['recommendations'].append(
                "Network is over 90% utilized. Consider adding more compute providers."
            )
        elif current_metrics['network_utilization'] > 75:
            health_report['overall_health'] = 'WARNING'
            health_report['recommendations'].append(
                "Network utilization is high. Monitor compute provider capacity."
            )

        if current_metrics['active_providers'] < 5:
            health_report['recommendations'].append(
                "Low number of active providers. Attract more compute resources."
            )

        return health_report

class MonitoringDashboard:
    """
    Simple CLI-based monitoring dashboard
    """
    @staticmethod
    def display_network_metrics(monitoring_service: NetworkMonitoringService):
        """
        Display network metrics in a formatted CLI output
        """
        metrics = monitoring_service.get_current_network_metrics()
        health_report = monitoring_service.generate_network_health_report()

        print("\n===== DAICN Network Monitoring Dashboard =====")
        print(f"Overall Network Health: {health_report['overall_health']}")
        print("\n--- Network Metrics ---")
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("\n--- Health Indicators ---")
        for key, value in health_report['health_indicators'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("\n--- Recommendations ---")
        for rec in health_report.get('recommendations', ['No current recommendations']):
            print(f"- {rec}")

        print("\n=======================================")

    @staticmethod
    async def start_dashboard_monitoring(
        monitoring_service: NetworkMonitoringService, 
        update_interval: int = 60
    ):
        """
        Start a simple monitoring dashboard that updates periodically
        """
        while True:
            MonitoringDashboard.display_network_metrics(monitoring_service)
            await asyncio.sleep(update_interval)
