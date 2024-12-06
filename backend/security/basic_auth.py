import uuid
import hashlib
import logging
from typing import Dict, Optional, Any
from enum import Enum, auto

class UserRole(Enum):
    """Basic user roles for access control"""
    GUEST = auto()
    USER = auto()
    PROVIDER = auto()
    ADMIN = auto()

class BasicAuthenticationManager:
    """
    Minimal viable authentication system with basic security features
    
    Key Security Considerations:
    - Password hashing
    - Basic role-based access control
    - Minimal user information storage
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize authentication manager
        
        Args:
            log_level (int): Logging level for the authentication system
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # In-memory user storage (replace with database in production)
        self.users: Dict[str, Dict[str, Any]] = {}
        
    def _hash_password(self, password: str) -> str:
        """
        Create a secure hash of the password
        
        Args:
            password (str): Plain text password
        
        Returns:
            str: Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(
        self, 
        username: str, 
        password: str, 
        email: Optional[str] = None,
        role: UserRole = UserRole.USER
    ) -> Dict[str, Any]:
        """
        Register a new user in the system
        
        Args:
            username (str): Unique username
            password (str): User password
            email (Optional[str]): Optional email address
            role (UserRole): User role in the system
        
        Returns:
            Dict[str, Any]: Registration result
        """
        if username in self.users:
            self.logger.warning(f"User {username} already exists")
            return {
                'status': 'error',
                'message': 'Username already exists'
            }
        
        user_id = str(uuid.uuid4())
        
        self.users[username] = {
            'user_id': user_id,
            'username': username,
            'password_hash': self._hash_password(password),
            'email': email,
            'role': role,
            'created_at': uuid.uuid1().time
        }
        
        self.logger.info(f"Registered new user: {username}")
        
        return {
            'status': 'success',
            'user_id': user_id,
            'username': username
        }
    
    def authenticate(
        self, 
        username: str, 
        password: str
    ) -> Dict[str, Any]:
        """
        Authenticate a user
        
        Args:
            username (str): Username
            password (str): Password
        
        Returns:
            Dict[str, Any]: Authentication result
        """
        if username not in self.users:
            self.logger.warning(f"Authentication attempt for non-existent user: {username}")
            return {
                'status': 'error',
                'message': 'Invalid username or password'
            }
        
        user = self.users[username]
        
        if user['password_hash'] != self._hash_password(password):
            self.logger.warning(f"Failed authentication attempt for user: {username}")
            return {
                'status': 'error',
                'message': 'Invalid username or password'
            }
        
        self.logger.info(f"Successful authentication for user: {username}")
        
        return {
            'status': 'success',
            'user_id': user['user_id'],
            'username': username,
            'role': user['role']
        }
    
    def get_user_role(self, username: str) -> Optional[UserRole]:
        """
        Retrieve user role
        
        Args:
            username (str): Username
        
        Returns:
            Optional[UserRole]: User role or None if user not found
        """
        user = self.users.get(username)
        return user['role'] if user else None
    
    def __len__(self) -> int:
        """
        Get number of registered users
        
        Returns:
            int: Number of registered users
        """
        return len(self.users)

def main():
    """
    Demonstration of Basic Authentication System
    """
    auth_manager = BasicAuthenticationManager()
    
    # Register users
    auth_manager.register_user('alice', 'password123', 'alice@example.com', UserRole.USER)
    auth_manager.register_user('bob', 'securepass', 'bob@example.com', UserRole.PROVIDER)
    auth_manager.register_user('admin', 'adminpass', 'admin@example.com', UserRole.ADMIN)
    
    # Authentication tests
    print("Alice Authentication:", auth_manager.authenticate('alice', 'password123'))
    print("Bob Authentication:", auth_manager.authenticate('bob', 'securepass'))
    print("Admin Authentication:", auth_manager.authenticate('admin', 'adminpass'))
    
    # Failed authentication test
    print("Failed Authentication:", auth_manager.authenticate('alice', 'wrongpassword'))

if __name__ == '__main__':
    main()
