import numpy as np
from typing import List, Union
import logging
import secrets

class HomomorphicEncryptionUtility:
    """
    Simplified Homomorphic Encryption Utility
    Provides basic homomorphic encryption capabilities for secure computation
    """
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize homomorphic encryption utility
        
        Args:
            key_size (int): Size of encryption keys
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.key_size = key_size
        self.public_key = self._generate_public_key()
        self.private_key = self._generate_private_key()
        
        self.logger.info(f"Homomorphic Encryption initialized with {key_size}-bit keys")

    def _generate_public_key(self):
        """
        Generate a public key for encryption
        
        Returns:
            dict: Public key components
        """
        # Simplified public key generation
        return {
            'modulus': secrets.randbits(self.key_size),
            'generator': secrets.randbits(self.key_size // 2)
        }

    def _generate_private_key(self):
        """
        Generate a private key for decryption
        
        Returns:
            int: Private key
        """
        return secrets.randbits(self.key_size)

    def encrypt(self, data: Union[int, float, List[float]]) -> dict:
        """
        Encrypt data using homomorphic encryption
        
        Args:
            data (int, float, or List[float]): Data to encrypt
        
        Returns:
            dict: Encrypted data representation
        """
        try:
            def _encrypt_single(value):
                # Simplified encryption mechanism
                r = secrets.randbits(self.key_size // 4)
                encrypted_value = (
                    (pow(self.public_key['generator'], value, self.public_key['modulus']) * 
                     pow(r, self.public_key['modulus'], self.public_key['modulus'])) 
                    % self.public_key['modulus']
                )
                return encrypted_value

            if isinstance(data, (int, float)):
                return {'type': 'scalar', 'value': _encrypt_single(data)}
            elif isinstance(data, list):
                return {
                    'type': 'vector', 
                    'values': [_encrypt_single(val) for val in data]
                }
            else:
                raise ValueError("Unsupported data type for encryption")
        
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise

    def decrypt(self, encrypted_data: dict) -> Union[int, float, List[float]]:
        """
        Decrypt homomorphically encrypted data
        
        Args:
            encrypted_data (dict): Encrypted data representation
        
        Returns:
            Decrypted data (int, float, or List[float])
        """
        try:
            def _decrypt_single(encrypted_value):
                # Simplified decryption mechanism
                decrypted = pow(
                    encrypted_value, 
                    self.private_key, 
                    self.public_key['modulus']
                )
                return decrypted

            if encrypted_data['type'] == 'scalar':
                return _decrypt_single(encrypted_data['value'])
            elif encrypted_data['type'] == 'vector':
                return [_decrypt_single(val) for val in encrypted_data['values']]
            else:
                raise ValueError("Invalid encrypted data format")
        
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise

    def compute_encrypted_sum(self, encrypted_values: List[dict]) -> dict:
        """
        Perform homomorphic addition on encrypted values
        
        Args:
            encrypted_values (List[dict]): List of encrypted values
        
        Returns:
            dict: Encrypted sum
        """
        try:
            if not all(val['type'] == encrypted_values[0]['type'] for val in encrypted_values):
                raise ValueError("All values must be of the same type")

            if encrypted_values[0]['type'] == 'scalar':
                encrypted_sum = {
                    'type': 'scalar',
                    'value': np.prod([val['value'] for val in encrypted_values]) % self.public_key['modulus']
                }
            elif encrypted_values[0]['type'] == 'vector':
                encrypted_sum = {
                    'type': 'vector',
                    'values': [
                        np.prod([val['values'][i] for val in encrypted_values]) % self.public_key['modulus']
                        for i in range(len(encrypted_values[0]['values']))
                    ]
                }
            
            return encrypted_sum
        
        except Exception as e:
            self.logger.error(f"Encrypted computation error: {e}")
            raise

def main():
    # Demonstration of homomorphic encryption
    he = HomomorphicEncryptionUtility()
    
    # Encrypt scalar values
    x = 10
    y = 20
    
    encrypted_x = he.encrypt(x)
    encrypted_y = he.encrypt(y)
    
    # Perform encrypted computation
    encrypted_sum = he.compute_encrypted_sum([encrypted_x, encrypted_y])
    
    # Decrypt result
    decrypted_sum = he.decrypt(encrypted_sum)
    
    print(f"Original values: {x}, {y}")
    print(f"Encrypted sum: {encrypted_sum}")
    print(f"Decrypted sum: {decrypted_sum}")
    
    # Encrypt vector
    vector = [1, 2, 3, 4, 5]
    encrypted_vector = he.encrypt(vector)
    
    # Compute encrypted vector sum
    encrypted_vector_sum = he.compute_encrypted_sum([encrypted_vector])
    decrypted_vector_sum = he.decrypt(encrypted_vector_sum)
    
    print(f"Original vector: {vector}")
    print(f"Decrypted vector sum: {decrypted_vector_sum}")

if __name__ == '__main__':
    main()
