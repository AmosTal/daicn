import os
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import logging
from typing import Dict, Any, Optional

class EthereumBlockchainInterface:
    def __init__(self, 
                 provider_url: str = 'https://goerli.infura.io/v3/YOUR_INFURA_PROJECT_ID',
                 contract_address: Optional[str] = None,
                 contract_abi_path: Optional[str] = None):
        """
        Initialize Ethereum blockchain interface
        
        Args:
            provider_url (str): Ethereum network provider URL
            contract_address (str, optional): Deployed contract address
            contract_abi_path (str, optional): Path to contract ABI file
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Connect to Ethereum network
        try:
            self.w3 = Web3(Web3.HTTPProvider(provider_url))
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Validate connection
            if not self.w3.isConnected():
                raise ConnectionError("Failed to connect to Ethereum network")
            
            self.logger.info("Successfully connected to Ethereum network")
        except Exception as e:
            self.logger.error(f"Ethereum connection error: {e}")
            raise
        
        # Load contract if provided
        self.contract = None
        if contract_address and contract_abi_path:
            self.load_contract(contract_address, contract_abi_path)

    def load_contract(self, contract_address: str, contract_abi_path: str):
        """
        Load a specific smart contract
        
        Args:
            contract_address (str): Deployed contract address
            contract_abi_path (str): Path to contract ABI file
        """
        try:
            # Load contract ABI
            with open(contract_abi_path, 'r') as abi_file:
                contract_abi = json.load(abi_file)
            
            # Create contract instance
            self.contract = self.w3.eth.contract(
                address=self.w3.toChecksumAddress(contract_address), 
                abi=contract_abi
            )
            
            self.logger.info(f"Loaded contract at {contract_address}")
        except Exception as e:
            self.logger.error(f"Contract loading error: {e}")
            raise

    def deploy_contract(self, 
                        contract_bytecode: str, 
                        contract_abi: Dict[str, Any], 
                        deployer_address: str, 
                        deployer_private_key: str,
                        constructor_args: tuple = ()):
        """
        Deploy a new smart contract
        
        Args:
            contract_bytecode (str): Compiled contract bytecode
            contract_abi (dict): Contract ABI
            deployer_address (str): Address deploying the contract
            deployer_private_key (str): Private key for signing transaction
            constructor_args (tuple): Arguments for contract constructor
        
        Returns:
            str: Deployed contract address
        """
        try:
            # Create contract factory
            contract = self.w3.eth.contract(
                abi=contract_abi, 
                bytecode=contract_bytecode
            )
            
            # Get transaction count (nonce)
            nonce = self.w3.eth.getTransactionCount(deployer_address)
            
            # Prepare contract deployment transaction
            transaction = contract.constructor(*constructor_args).buildTransaction({
                'chainId': 5,  # Goerli testnet
                'gas': 2000000,
                'gasPrice': self.w3.toWei('50', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.signTransaction(
                transaction, 
                private_key=deployer_private_key
            )
            
            # Send transaction
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
            
            self.logger.info(f"Contract deployed at: {tx_receipt.contractAddress}")
            return tx_receipt.contractAddress
        
        except Exception as e:
            self.logger.error(f"Contract deployment error: {e}")
            raise

    def call_contract_function(self, 
                                function_name: str, 
                                function_args: tuple = (), 
                                sender_address: Optional[str] = None):
        """
        Call a read-only contract function
        
        Args:
            function_name (str): Name of contract function to call
            function_args (tuple): Arguments for the function
            sender_address (str, optional): Address calling the function
        
        Returns:
            Any: Function return value
        """
        if not self.contract:
            raise ValueError("No contract loaded")
        
        try:
            contract_function = getattr(self.contract.functions, function_name)
            
            if sender_address:
                result = contract_function(*function_args).call({'from': sender_address})
            else:
                result = contract_function(*function_args).call()
            
            return result
        
        except Exception as e:
            self.logger.error(f"Contract function call error: {e}")
            raise

    def send_contract_transaction(self, 
                                  function_name: str, 
                                  sender_address: str, 
                                  private_key: str, 
                                  function_args: tuple = ()):
        """
        Send a transaction to modify contract state
        
        Args:
            function_name (str): Name of contract function to call
            sender_address (str): Address sending the transaction
            private_key (str): Private key for signing transaction
            function_args (tuple): Arguments for the function
        
        Returns:
            str: Transaction hash
        """
        if not self.contract:
            raise ValueError("No contract loaded")
        
        try:
            # Get transaction count (nonce)
            nonce = self.w3.eth.getTransactionCount(sender_address)
            
            # Prepare contract transaction
            contract_function = getattr(self.contract.functions, function_name)
            transaction = contract_function(*function_args).buildTransaction({
                'chainId': 5,  # Goerli testnet
                'gas': 200000,
                'gasPrice': self.w3.toWei('50', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.signTransaction(
                transaction, 
                private_key=private_key
            )
            
            # Send transaction
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
            
            self.logger.info(f"Transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
        
        except Exception as e:
            self.logger.error(f"Contract transaction error: {e}")
            raise

def main():
    # Example usage
    try:
        # Initialize blockchain interface
        eth_interface = EthereumBlockchainInterface(
            provider_url='https://goerli.infura.io/v3/YOUR_PROJECT_ID'
        )
        
        # Example: Deploy a mock contract (requires bytecode and ABI)
        # contract_address = eth_interface.deploy_contract(
        #     contract_bytecode='0x...',
        #     contract_abi=[...],
        #     deployer_address='0x...',
        #     deployer_private_key='0x...'
        # )
        
        print("Ethereum Blockchain Interface initialized successfully")
    
    except Exception as e:
        print(f"Initialization failed: {e}")

if __name__ == '__main__':
    main()
