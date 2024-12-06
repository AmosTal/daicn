import asyncio
import logging
from typing import Dict, Any, List
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class NotificationService:
    def __init__(self, 
                 smtp_host: str = 'smtp.gmail.com', 
                 smtp_port: int = 587, 
                 sender_email: str = None, 
                 sender_password: str = None):
        """
        Initialize notification service with SMTP configuration
        
        :param smtp_host: SMTP server hostname
        :param smtp_port: SMTP server port
        :param sender_email: Email address used to send notifications
        :param sender_password: Password for the sender email
        """
        self.logger = logging.getLogger(__name__)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    async def send_email_notification(
        self, 
        recipient: str, 
        subject: str, 
        body: str, 
        html_body: str = None
    ) -> bool:
        """
        Send an email notification asynchronously
        
        :param recipient: Email address of the recipient
        :param subject: Email subject
        :param body: Plain text email body
        :param html_body: Optional HTML email body
        :return: Boolean indicating successful email send
        """
        if not all([self.sender_email, self.sender_password]):
            self.logger.warning("SMTP credentials not configured")
            return False

        try:
            message = MIMEMultipart('alternative')
            message['From'] = self.sender_email
            message['To'] = recipient
            message['Subject'] = subject

            # Attach plain text and HTML parts
            message.attach(MIMEText(body, 'plain'))
            if html_body:
                message.attach(MIMEText(html_body, 'html'))

            async with aiosmtplib.SMTP(
                hostname=self.smtp_host, 
                port=self.smtp_port
            ) as smtp:
                await smtp.starttls()
                await smtp.login(self.sender_email, self.sender_password)
                await smtp.send_message(message)

            self.logger.info(f"Email sent to {recipient}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False

    async def send_slack_notification(
        self, 
        webhook_url: str, 
        message: str, 
        channel: str = None, 
        username: str = 'DAICN Notification Bot'
    ) -> bool:
        """
        Send a Slack notification using a webhook
        
        :param webhook_url: Slack webhook URL
        :param message: Notification message
        :param channel: Optional channel to post in
        :param username: Bot username
        :return: Boolean indicating successful notification
        """
        import aiohttp

        payload = {
            'text': message,
            'username': username
        }
        
        if channel:
            payload['channel'] = channel

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Slack notification sent successfully")
                        return True
                    else:
                        self.logger.error(f"Failed to send Slack notification: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
            return False

    async def send_network_health_alert(
        self, 
        network_health_data: Dict[str, Any], 
        recipients: List[str] = None
    ):
        """
        Send network health alerts based on predefined thresholds
        
        :param network_health_data: Network health metrics dictionary
        :param recipients: List of email recipients
        """
        health_score = network_health_data.get('health_score', 0)
        
        # Define alert thresholds
        if health_score < 50:
            alert_level = 'CRITICAL'
            alert_color = '🔴'
        elif health_score < 70:
            alert_level = 'WARNING'
            alert_color = '🟠'
        else:
            return  # No alert needed
        
        # Construct alert message
        email_body = f"""
        {alert_color} DAICN Network Health Alert - {alert_level} {alert_color}
        
        Network Health Score: {health_score}%
        
        Details:
        - Total Providers: {network_health_data.get('total_providers', 0)}
        - Active Providers: {network_health_data.get('active_providers', 0)}
        - Total Tasks: {network_health_data.get('total_tasks', 0)}
        - Completed Tasks: {network_health_data.get('completed_tasks', 0)}
        
        Recommended Actions:
        1. Review provider performance
        2. Check task allocation mechanisms
        3. Investigate potential network bottlenecks
        """
        
        # Send email alerts
        if recipients:
            for recipient in recipients:
                await self.send_email_notification(
                    recipient, 
                    f"{alert_color} DAICN Network Health Alert - {alert_level}", 
                    email_body
                )
        
        # Optional: Send Slack alert (configure webhook URL)
        # await self.send_slack_notification(
        #     webhook_url='YOUR_SLACK_WEBHOOK_URL',
        #     message=email_body
        # )

    @classmethod
    async def run_example_notifications(cls):
        """
        Example method to demonstrate notification capabilities
        """
        # Initialize with your SMTP and Slack credentials
        notification_service = cls(
            smtp_host='smtp.gmail.com',
            smtp_port=587,
            sender_email='your_email@gmail.com',
            sender_password='your_app_password'
        )

        # Example email notification
        await notification_service.send_email_notification(
            recipient='recipient@example.com',
            subject='DAICN Network Notification',
            body='This is a test notification from DAICN',
            html_body='<h1>DAICN Network Notification</h1><p>This is a test notification</p>'
        )

        # Example network health alert
        network_health_data = {
            'health_score': 45,
            'total_providers': 100,
            'active_providers': 60,
            'total_tasks': 1000,
            'completed_tasks': 750
        }
        await notification_service.send_network_health_alert(
            network_health_data, 
            recipients=['admin1@example.com', 'admin2@example.com']
        )
