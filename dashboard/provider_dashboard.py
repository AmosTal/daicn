import flask
from flask_login import login_required, current_user
from backend.models import Provider, Task
from backend.security.authentication import SecureAuthentication

class ProviderDashboard:
    def __init__(self, app, db):
        self.app = app
        self.db = db
        self.auth = SecureAuthentication(app.config['SECRET_KEY'])
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/provider/dashboard')
        @login_required
        def provider_dashboard():
            """Main provider dashboard view."""
            provider = Provider.query.get(current_user.id)
            
            # Compute resource metrics
            total_tasks = Task.query.filter_by(provider_id=provider.id).count()
            completed_tasks = Task.query.filter_by(
                provider_id=provider.id, 
                status='COMPLETED'
            ).count()
            
            # Earnings calculation
            total_earnings = sum(
                task.reward for task in 
                Task.query.filter_by(provider_id=provider.id, status='COMPLETED')
            )
            
            # Performance metrics
            performance_score = self.calculate_performance_score(provider)
            
            return flask.render_template(
                'provider_dashboard.html', 
                provider=provider,
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                total_earnings=total_earnings,
                performance_score=performance_score
            )
        
        @self.app.route('/provider/tasks')
        @login_required
        def provider_tasks():
            """View of available and past tasks."""
            provider = Provider.query.get(current_user.id)
            
            available_tasks = Task.query.filter_by(status='AVAILABLE').all()
            my_tasks = Task.query.filter_by(provider_id=provider.id).all()
            
            return flask.render_template(
                'provider_tasks.html',
                available_tasks=available_tasks,
                my_tasks=my_tasks
            )
        
        @self.app.route('/provider/accept_task/<int:task_id>', methods=['POST'])
        @login_required
        def accept_task(task_id):
            """Accept a new computational task."""
            task = Task.query.get(task_id)
            provider = Provider.query.get(current_user.id)
            
            if task and task.status == 'AVAILABLE':
                # Check provider's capacity
                if provider.can_accept_task(task):
                    task.provider_id = provider.id
                    task.status = 'IN_PROGRESS'
                    self.db.session.commit()
                    flask.flash('Task accepted successfully!')
                else:
                    flask.flash('Insufficient compute resources.')
            
            return flask.redirect(flask.url_for('provider_tasks'))

    def calculate_performance_score(self, provider):
        """Calculate provider's performance score."""
        completed_tasks = Task.query.filter_by(
            provider_id=provider.id, 
            status='COMPLETED'
        )
        
        total_tasks = Task.query.filter_by(provider_id=provider.id)
        
        # Complex performance calculation
        performance_metrics = {
            'completion_rate': len(completed_tasks) / len(total_tasks) if total_tasks else 0,
            'average_task_time': self.calculate_avg_task_time(provider),
            'reputation_score': provider.reputation_score
        }
        
        # Weighted performance score
        performance_score = (
            performance_metrics['completion_rate'] * 0.4 +
            (1 - performance_metrics['average_task_time']) * 0.3 +
            performance_metrics['reputation_score'] * 0.3
        ) * 100
        
        return round(performance_score, 2)

    def calculate_avg_task_time(self, provider):
        """Calculate average task completion time."""
        # Implementation depends on your specific task tracking
        return 0.5  # Placeholder
