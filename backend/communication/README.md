# Inter-Component Communication Protocol

## Overview
This module provides a robust, flexible communication protocol for inter-component messaging in the Decentralized AI Computation Network (DAICN).

## Key Features
- Standardized message structure
- Asynchronous communication
- Message type enumeration
- Automatic message tracking
- Configurable message handlers
- Built-in retry and error handling

## Components
- `MessageType`: Enum defining communication message types
- `CommunicationMessage`: Standardized message structure
- `InterComponentCommunicationProtocol`: Core communication management

## Usage Example
```python
from inter_component_protocol import (
    InterComponentCommunicationProtocol, 
    MessageType, 
    CommunicationMessage
)

# Create a communication protocol instance
task_allocator = InterComponentCommunicationProtocol('task_allocator')

# Define a message handler
def task_allocation_handler(message):
    print(f"Received task: {message.payload}")

# Register the handler
task_allocator.register_message_handler(
    MessageType.TASK_ALLOCATION, 
    task_allocation_handler
)

# Create and send a message
sample_message = CommunicationMessage(
    message_type=MessageType.TASK_ALLOCATION,
    sender='orchestration_manager',
    recipient='task_allocator',
    payload={'task_details': 'complex_computation'}
)

# Send the message
await task_allocator.send_message(sample_message)
```

## Future Improvements
- Persistent message queuing
- Enhanced encryption
- Distributed message routing
- Advanced error recovery
