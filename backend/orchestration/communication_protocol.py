import uuid
import time
import json
import logging
import threading
from typing import Dict, Any, List, Callable
import zmq
import msgpack
import nacl.secret
import nacl.utils

class CommunicationProtocol:
    """
    Advanced Inter-Component Communication Protocol
    
    Provides secure, efficient, and reliable communication between network components
    """
    
    def __init__(
        self, 
        host: str = '*', 
        pub_port: int = 5555, 
        sub_port: int = 5556,
        encryption_key: bytes = None
    ):
        """
        Initialize Communication Protocol
        
        Args:
            host (str): Binding host
            pub_port (int): Publishing port
            sub_port (int): Subscribing port
            encryption_key (bytes): Optional encryption key
        """
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # ZeroMQ context and sockets
        self.context = zmq.Context()
        
        # Publisher socket for broadcasting messages
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://{host}:{pub_port}")
        
        # Subscriber socket for receiving messages
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.bind(f"tcp://{host}:{sub_port}")
        
        # Message encryption
        self.encryption_key = encryption_key or nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
        self.encryption_box = nacl.secret.SecretBox(self.encryption_key)
        
        # Message tracking
        self.message_registry = {}
        self.registry_lock = threading.Lock()
        
        # Subscription management
        self.subscriptions = set()
        
        self.logger.info("Communication Protocol initialized")

    def publish_message(
        self, 
        topic: str, 
        message: Dict[str, Any], 
        require_confirmation: bool = False
    ) -> str:
        """
        Publish a message to a specific topic
        
        Args:
            topic (str): Message topic
            message (Dict[str, Any]): Message payload
            require_confirmation (bool): Require message delivery confirmation
        
        Returns:
            str: Message ID
        """
        try:
            # Generate unique message ID
            message_id = str(uuid.uuid4())
            
            # Prepare message payload
            message_payload = {
                'message_id': message_id,
                'topic': topic,
                'timestamp': time.time(),
                'payload': message,
                'require_confirmation': require_confirmation
            }
            
            # Serialize and encrypt message
            serialized_message = msgpack.packb(message_payload)
            encrypted_message = self.encryption_box.encrypt(serialized_message)
            
            # Publish encrypted message
            self.publisher.send_multipart([
                topic.encode('utf-8'), 
                encrypted_message
            ])
            
            # Track message if confirmation required
            if require_confirmation:
                with self.registry_lock:
                    self.message_registry[message_id] = {
                        'status': 'sent',
                        'timestamp': time.time()
                    }
            
            self.logger.info(f"Message {message_id} published to topic {topic}")
            return message_id
        
        except Exception as e:
            self.logger.error(f"Message publication error: {e}")
            raise

    def subscribe_to_topic(self, topic: str):
        """
        Subscribe to a specific topic
        
        Args:
            topic (str): Topic to subscribe
        """
        try:
            self.subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
            self.subscriptions.add(topic)
            self.logger.info(f"Subscribed to topic: {topic}")
        
        except Exception as e:
            self.logger.error(f"Topic subscription error: {e}")
            raise

    def receive_messages(
        self, 
        callback: Callable[[Dict[str, Any]], None] = None,
        topics: List[str] = None
    ):
        """
        Receive messages from subscribed topics
        
        Args:
            callback (Callable): Optional message processing callback
            topics (List[str]): Optional list of topics to listen
        """
        def message_handler():
            while True:
                try:
                    # Receive multipart message
                    topic, encrypted_message = self.subscriber.recv_multipart()
                    topic = topic.decode('utf-8')
                    
                    # Decrypt message
                    decrypted_message = self.encryption_box.decrypt(encrypted_message)
                    message_payload = msgpack.unpackb(decrypted_message)
                    
                    # Filter by specified topics
                    if topics and topic not in topics:
                        continue
                    
                    # Process message
                    if callback:
                        callback(message_payload)
                    
                    # Send confirmation if required
                    if message_payload.get('require_confirmation'):
                        self._send_message_confirmation(message_payload['message_id'])
                
                except Exception as e:
                    self.logger.error(f"Message reception error: {e}")
        
        # Start message reception in a separate thread
        message_thread = threading.Thread(target=message_handler, daemon=True)
        message_thread.start()
        
        return message_thread

    def _send_message_confirmation(self, message_id: str):
        """
        Send message delivery confirmation
        
        Args:
            message_id (str): Message to confirm
        """
        try:
            confirmation_message = {
                'message_id': message_id,
                'status': 'confirmed',
                'timestamp': time.time()
            }
            
            # Publish confirmation
            self.publish_message('confirmations', confirmation_message)
            
            self.logger.info(f"Confirmation sent for message {message_id}")
        
        except Exception as e:
            self.logger.error(f"Confirmation sending error: {e}")
            raise

    def track_message_confirmations(self):
        """
        Track message delivery confirmations
        """
        def confirmation_tracker():
            while True:
                try:
                    with self.registry_lock:
                        # Check and update message status
                        current_time = time.time()
                        for message_id, message_info in list(self.message_registry.items()):
                            if message_info['status'] == 'sent' and \
                               current_time - message_info['timestamp'] > 30:
                                # Mark as unconfirmed after 30 seconds
                                message_info['status'] = 'unconfirmed'
                
                # Wait before next check
                time.sleep(10)
        
        # Start confirmation tracking thread
        tracking_thread = threading.Thread(target=confirmation_tracker, daemon=True)
        tracking_thread.start()
        
        return tracking_thread

def example_message_handler(message: Dict[str, Any]):
    """
    Example message processing callback
    
    Args:
        message (Dict[str, Any]): Received message
    """
    print(f"Received message: {message}")

def main():
    # Create communication protocol instances
    protocol1 = CommunicationProtocol(host='localhost', pub_port=5555, sub_port=5556)
    protocol2 = CommunicationProtocol(host='localhost', pub_port=5556, sub_port=5555)
    
    # Subscribe to topics
    protocol2.subscribe_to_topic('task_allocation')
    protocol2.subscribe_to_topic('system_status')
    
    # Start message reception
    protocol2.receive_messages(
        callback=example_message_handler, 
        topics=['task_allocation', 'system_status']
    )
    
    # Start confirmation tracking
    protocol1.track_message_confirmations()
    
    # Publish example messages
    task_allocation_message = {
        'task_id': str(uuid.uuid4()),
        'provider_id': 'provider_001',
        'complexity': 'medium'
    }
    
    system_status_message = {
        'total_providers': 50,
        'active_tasks': 25,
        'network_health': 0.95
    }
    
    # Publish messages with confirmation
    protocol1.publish_message(
        'task_allocation', 
        task_allocation_message, 
        require_confirmation=True
    )
    
    protocol1.publish_message(
        'system_status', 
        system_status_message, 
        require_confirmation=True
    )
    
    # Keep main thread running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Communication Protocol demo stopped")

if __name__ == '__main__':
    main()
