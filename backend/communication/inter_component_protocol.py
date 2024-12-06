import uuid
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from enum import Enum, auto

class MessageType(Enum):
    """Enumeration of message types for inter-component communication"""
    TASK_ALLOCATION = auto()
    TASK_RESULT = auto()
    RESOURCE_REQUEST = auto()
    RESOURCE_RESPONSE = auto()
    SYSTEM_HEALTH = auto()
    ERROR_NOTIFICATION = auto()
    HEARTBEAT = auto()

class CommunicationMessage:
    """
    Standardized message structure for inter-component communication
    """
    def __init__(
        self, 
        message_type: MessageType, 
        sender: str, 
        recipient: str, 
        payload: Dict[str, Any],
        priority: int = 0
    ):
        """
        Initialize a communication message
        
        Args:
            message_type (MessageType): Type of message
            sender (str): Sending component identifier
            recipient (str): Receiving component identifier
            payload (Dict): Message payload
            priority (int): Message priority (optional)
        """
        self.id = str(uuid.uuid4())
        self.type = message_type
        self.sender = sender
        self.recipient = recipient
        self.payload = payload
        self.priority = priority
        self.timestamp = time.time()
        self.status = 'created'
        self.retries = 0
        self.max_retries = 3

    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps({
            'id': self.id,
            'type': self.type.name,
            'sender': self.sender,
            'recipient': self.recipient,
            'payload': self.payload,
            'priority': self.priority,
            'timestamp': self.timestamp
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'CommunicationMessage':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        message = cls(
            message_type=MessageType[data['type']],
            sender=data['sender'],
            recipient=data['recipient'],
            payload=data['payload'],
            priority=data.get('priority', 0)
        )
        message.id = data['id']
        message.timestamp = data['timestamp']
        return message

class InterComponentCommunicationProtocol:
    """
    Manages communication between different components of the distributed system
    """
    def __init__(self, component_id: str):
        """
        Initialize communication protocol
        
        Args:
            component_id (str): Unique identifier for this component
        """
        self.component_id = component_id
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.sent_messages: Dict[str, CommunicationMessage] = {}
        self.received_messages: Dict[str, CommunicationMessage] = {}
        
        # Logging setup
        self.logger = logging.getLogger(f'communication_{component_id}')
        self.logger.setLevel(logging.INFO)

    def register_message_handler(
        self, 
        message_type: MessageType, 
        handler: Callable[[CommunicationMessage], None]
    ):
        """
        Register a handler for a specific message type
        
        Args:
            message_type (MessageType): Message type to handle
            handler (Callable): Function to process the message
        """
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for {message_type}")

    async def send_message(
        self, 
        message: CommunicationMessage
    ) -> bool:
        """
        Send a message to another component
        
        Args:
            message (CommunicationMessage): Message to send
        
        Returns:
            bool: Whether message was sent successfully
        """
        try:
            # Simulate message sending (replace with actual network communication)
            self.sent_messages[message.id] = message
            message.status = 'sent'
            
            self.logger.info(
                f"Sending message {message.id} "
                f"from {message.sender} to {message.recipient}"
            )
            
            # Simulate potential network delay or failure
            await asyncio.sleep(0.1)  # Simulated network delay
            
            return True
        
        except Exception as e:
            self.logger.error(f"Message sending failed: {e}")
            message.status = 'failed'
            message.retries += 1
            
            if message.retries >= message.max_retries:
                self.logger.error(f"Max retries reached for message {message.id}")
                return False
            
            return False

    async def receive_message(
        self, 
        message: CommunicationMessage
    ):
        """
        Process an incoming message
        
        Args:
            message (CommunicationMessage): Received message
        """
        try:
            # Validate message
            if message.recipient != self.component_id:
                self.logger.warning(
                    f"Message {message.id} not intended for this component"
                )
                return
            
            # Store received message
            self.received_messages[message.id] = message
            message.status = 'received'
            
            # Find and execute appropriate handler
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message)
            else:
                self.logger.warning(
                    f"No handler registered for message type {message.type}"
                )
        
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            # Optionally send error notification

    async def heartbeat(self):
        """
        Periodically send heartbeat messages to maintain system connectivity
        """
        while True:
            heartbeat_msg = CommunicationMessage(
                message_type=MessageType.HEARTBEAT,
                sender=self.component_id,
                recipient='system_monitor',
                payload={
                    'timestamp': time.time(),
                    'status': 'active'
                }
            )
            
            await self.send_message(heartbeat_msg)
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds

def main():
    """Example usage demonstration"""
    async def example_usage():
        # Create communication protocol for a task allocation component
        task_allocator = InterComponentCommunicationProtocol('task_allocator')
        
        # Define a simple message handler
        def task_allocation_handler(message: CommunicationMessage):
            print(f"Received task allocation message: {message.payload}")
        
        # Register the handler
        task_allocator.register_message_handler(
            MessageType.TASK_ALLOCATION, 
            task_allocation_handler
        )
        
        # Create and send a sample message
        sample_message = CommunicationMessage(
            message_type=MessageType.TASK_ALLOCATION,
            sender='orchestration_manager',
            recipient='task_allocator',
            payload={
                'task_id': 'task_123',
                'complexity': 'high',
                'resource_requirements': {
                    'cpu': 4,
                    'memory': '16GB'
                }
            }
        )
        
        # Send the message
        await task_allocator.send_message(sample_message)
        
        # Simulate receiving the message
        await task_allocator.receive_message(sample_message)

    # Run the example
    asyncio.run(example_usage())

if __name__ == '__main__':
    main()
