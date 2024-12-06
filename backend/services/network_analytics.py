import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..models import ComputeProvider, ComputeTask, AIModel
from ..database import engine

class NetworkAnalyticsService:
    def __init__(self, db_session: Session):
        self.logger = logging.getLogger(__name__)
        self.db = db_session

    def compute_network_health_score(self) -> Dict[str, Any]:
        """
        Calculate an overall network health score
        """
        # Provider metrics
        total_providers = self.db.query(ComputeProvider).count()
        active_providers = self.db.query(ComputeProvider).filter(ComputeProvider.is_active == True).count()
        
        # Task metrics
        total_tasks = self.db.query(ComputeTask).count()
        completed_tasks = self.db.query(ComputeTask).filter(ComputeTask.status == 'completed').count()
        
        # Compute health score
        provider_health = active_providers / total_providers if total_providers > 0 else 0
        task_completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        # Weighted health score
        health_score = (0.6 * provider_health + 0.4 * task_completion_rate) * 100

        return {
            'total_providers': total_providers,
            'active_providers': active_providers,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'health_score': round(health_score, 2)
        }

    def analyze_provider_performance(self, time_window: int = 30) -> List[Dict[str, Any]]:
        """
        Analyze provider performance over a specified time window
        """
        cutoff_date = datetime.utcnow() - timedelta(days=time_window)
        
        # Aggregate provider performance
        providers_performance = (
            self.db.query(
                ComputeProvider.id,
                ComputeProvider.wallet_address,
                ComputeProvider.compute_power,
                ComputeProvider.reputation_score,
                func.count(ComputeTask.id).label('total_tasks'),
                func.sum(
                    func.case(
                        [(ComputeTask.status == 'completed', 1)], 
                        else_=0
                    )
                ).label('completed_tasks')
            )
            .outerjoin(ComputeTask, 
                and_(
                    ComputeTask.provider_id == ComputeProvider.id,
                    ComputeTask.created_at >= cutoff_date
                )
            )
            .group_by(
                ComputeProvider.id, 
                ComputeProvider.wallet_address, 
                ComputeProvider.compute_power,
                ComputeProvider.reputation_score
            )
            .all()
        )

        # Process and rank providers
        performance_data = []
        for provider in providers_performance:
            completion_rate = (provider.completed_tasks / provider.total_tasks) * 100 if provider.total_tasks > 0 else 0
            performance_score = (
                0.4 * completion_rate + 
                0.3 * provider.reputation_score + 
                0.3 * (provider.compute_power / 1000)  # Normalize compute power
            )

            performance_data.append({
                'provider_id': provider.id,
                'wallet_address': provider.wallet_address,
                'compute_power': provider.compute_power,
                'reputation_score': provider.reputation_score,
                'total_tasks': provider.total_tasks,
                'completed_tasks': provider.completed_tasks,
                'completion_rate': round(completion_rate, 2),
                'performance_score': round(performance_score, 2)
            })

        # Sort by performance score
        return sorted(performance_data, key=lambda x: x['performance_score'], reverse=True)

    def task_complexity_analysis(self, time_window: int = 30) -> Dict[str, Any]:
        """
        Analyze task complexity and resource utilization
        """
        cutoff_date = datetime.utcnow() - timedelta(days=time_window)
        
        # Analyze task types and compute requirements
        task_complexity = (
            self.db.query(
                ComputeTask.task_type,
                func.count(ComputeTask.id).label('total_tasks'),
                func.avg(ComputeTask.compute_units_required).label('avg_compute_units'),
                func.sum(ComputeTask.reward_amount).label('total_rewards')
            )
            .filter(ComputeTask.created_at >= cutoff_date)
            .group_by(ComputeTask.task_type)
            .all()
        )

        # Process complexity data
        complexity_data = []
        total_tasks = 0
        for task in task_complexity:
            total_tasks += task.total_tasks
            complexity_data.append({
                'task_type': task.task_type,
                'total_tasks': task.total_tasks,
                'task_percentage': round((task.total_tasks / total_tasks) * 100, 2) if total_tasks > 0 else 0,
                'avg_compute_units': round(task.avg_compute_units, 2),
                'total_rewards': round(task.total_rewards, 2)
            })

        return {
            'time_window_days': time_window,
            'total_tasks': total_tasks,
            'task_complexity': complexity_data
        }

    def generate_predictive_demand_forecast(self, prediction_window: int = 7) -> Dict[str, Any]:
        """
        Generate a predictive forecast of network demand
        """
        # Historical task data
        historical_tasks = (
            self.db.query(
                func.date_trunc('day', ComputeTask.created_at).label('task_date'),
                func.count(ComputeTask.id).label('daily_tasks'),
                func.sum(ComputeTask.compute_units_required).label('total_compute_units')
            )
            .group_by('task_date')
            .order_by('task_date')
            .limit(30)  # Last 30 days of data
            .all()
        )

        # Convert to pandas for time series analysis
        df = pd.DataFrame(
            [(row.task_date, row.daily_tasks, row.total_compute_units) for row in historical_tasks],
            columns=['date', 'daily_tasks', 'total_compute_units']
        )

        # Simple moving average forecast
        df['tasks_ma'] = df['daily_tasks'].rolling(window=7).mean()
        df['compute_units_ma'] = df['total_compute_units'].rolling(window=7).mean()

        # Predict next 7 days
        forecast = {
            'daily_tasks': {
                'historical_mean': df['daily_tasks'].mean(),
                'historical_std': df['daily_tasks'].std(),
                'forecast_mean': df['tasks_ma'].iloc[-1]
            },
            'compute_units': {
                'historical_mean': df['total_compute_units'].mean(),
                'historical_std': df['total_compute_units'].std(),
                'forecast_mean': df['compute_units_ma'].iloc[-1]
            }
        }

        return {
            'prediction_window_days': prediction_window,
            'forecast': forecast
        }

    def export_network_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive network report
        """
        return {
            'timestamp': datetime.utcnow(),
            'network_health': self.compute_network_health_score(),
            'provider_performance': self.analyze_provider_performance(),
            'task_complexity': self.task_complexity_analysis(),
            'demand_forecast': self.generate_predictive_demand_forecast()
        }

    async def periodic_network_analysis(self, interval_hours: int = 24):
        """
        Periodically run network-wide analysis
        """
        while True:
            try:
                report = self.export_network_report()
                self.logger.info("Generated periodic network analysis report")
                
                # Optional: Store report in database or send notifications
                # You could extend this to trigger alerts, update dashboards, etc.
                
                await asyncio.sleep(interval_hours * 3600)
            
            except Exception as e:
                self.logger.error(f"Error in periodic network analysis: {e}")
                await asyncio.sleep(interval_hours * 3600)

    @classmethod
    def run_example_analysis(cls):
        """
        Example method to demonstrate analytics capabilities
        """
        from ..database import SessionLocal
        
        # Create a database session
        db = SessionLocal()
        
        try:
            # Initialize analytics service
            analytics_service = cls(db)
            
            # Generate and print reports
            print("Network Health Score:")
            print(analytics_service.compute_network_health_score())
            
            print("\nProvider Performance:")
            for provider in analytics_service.analyze_provider_performance():
                print(provider)
            
            print("\nTask Complexity Analysis:")
            print(analytics_service.task_complexity_analysis())
            
            print("\nDemand Forecast:")
            print(analytics_service.generate_predictive_demand_forecast())
            
            print("\nComprehensive Network Report:")
            print(analytics_service.export_network_report())
        
        finally:
            db.close()
