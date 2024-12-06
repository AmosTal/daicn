import secrets
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecureAuthentication:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.encryption_key = self._generate_encryption_key()

    def _generate_encryption_key(self):
        """Generate a secure encryption key using PBKDF2."""
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
        return key

    def generate_token(self, user_id):
        """Generate a secure, time-bound authentication token."""
        timestamp = int(time.time())
        payload = f"{user_id}:{timestamp}"
        signature = hashlib.sha256(
            f"{payload}:{self.secret_key}".encode()
        ).hexdigest()
        return base64.urlsafe_b64encode(
            f"{payload}:{signature}".encode()
        ).decode()

    def validate_token(self, token, max_age=3600):
        """Validate an authentication token."""
        try:
            decoded_token = base64.urlsafe_b64decode(token.encode()).decode()
            user_id, timestamp, signature = decoded_token.rsplit(':', 2)
            
            # Check token age
            current_time = int(time.time())
            if current_time - int(timestamp) > max_age:
                return False

            # Verify signature
            expected_signature = hashlib.sha256(
                f"{user_id}:{timestamp}:{self.secret_key}".encode()
            ).hexdigest()

            return signature == expected_signature
        except Exception:
            return False

    def encrypt_data(self, data):
        """Encrypt sensitive data."""
        f = Fernet(self.encryption_key)
        return f.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data."""
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()

    def generate_mfa_code(self):
        """Generate a multi-factor authentication code."""
        return ''.join(secrets.choice('0123456789') for _ in range(6))

# Example usage
def main():
    auth = SecureAuthentication('your_secret_key')
    
    # Generate token for a user
    user_token = auth.generate_token('user123')
    print(f"Generated Token: {user_token}")
    
    # Validate token
    is_valid = auth.validate_token(user_token)
    print(f"Token Valid: {is_valid}")
    
    # Encrypt and decrypt data
    sensitive_data = "Confidential AI Compute Details"
    encrypted = auth.encrypt_data(sensitive_data)
    decrypted = auth.decrypt_data(encrypted)
    
    print(f"Original: {sensitive_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")

if __name__ == '__main__':
    main()
