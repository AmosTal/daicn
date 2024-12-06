import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
import uuid

class DAICNTokenEconomics:
    """
    Token Economic Model for Decentralized AI Computation Network
    Simulates token dynamics, provider incentives, and network economics
    """
    
    def __init__(self, 
                 total_supply: int = 100_000_000,  # 100 million tokens
                 initial_price: float = 0.10):
        """
        Initialize token economic model
        
        Args:
            total_supply (int): Total token supply
            initial_price (float): Initial token price
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Token parameters
        self.total_supply = total_supply
        self.circulating_supply = 0
        self.initial_price = initial_price
        
        # Economic simulation parameters
        self.token_price_history = [initial_price]
        self.network_metrics = {
            'total_providers': 0,
            'total_compute_power': 0,
            'total_tasks_processed': 0,
            'network_utilization': 0
        }
        
        # Provider incentive model
        self.provider_rewards = {}
        
        self.logger.info("DAICN Token Economic Model initialized")

    def simulate_token_distribution(self, 
                                    team_allocation: float = 0.2,
                                    provider_rewards: float = 0.4,
                                    community_pool: float = 0.3,
                                    initial_sale: float = 0.1):
        """
        Simulate initial token distribution
        
        Args:
            team_allocation (float): Percentage for team and advisors
            provider_rewards (float): Percentage for provider incentives
            community_pool (float): Percentage for community development
            initial_sale (float): Percentage for initial token sale
        """
        allocations = {
            'Team & Advisors': self.total_supply * team_allocation,
            'Provider Rewards': self.total_supply * provider_rewards,
            'Community Pool': self.total_supply * community_pool,
            'Initial Sale': self.total_supply * initial_sale
        }
        
        self.circulating_supply = allocations['Initial Sale']
        
        self.logger.info("Token Distribution Simulation:")
        for category, amount in allocations.items():
            self.logger.info(f"{category}: {amount:,} tokens")
        
        return allocations

    def calculate_provider_rewards(self, 
                                   compute_power: float, 
                                   tasks_processed: int, 
                                   network_contribution: float):
        """
        Calculate rewards for compute providers
        
        Args:
            compute_power (float): Provider's computational resources
            tasks_processed (int): Number of tasks completed
            network_contribution (float): Overall network contribution
        
        Returns:
            float: Calculated token rewards
        """
        base_reward = 10  # Base reward per task
        
        # Compute reward factors
        compute_factor = np.log1p(compute_power)
        task_factor = np.log1p(tasks_processed)
        contribution_factor = network_contribution
        
        # Calculate total reward
        total_reward = base_reward * compute_factor * task_factor * contribution_factor
        
        # Store provider reward
        provider_id = str(uuid.uuid4())
        self.provider_rewards[provider_id] = {
            'compute_power': compute_power,
            'tasks_processed': tasks_processed,
            'reward': total_reward
        }
        
        return total_reward

    def simulate_network_economics(self, 
                                   num_providers: int = 100, 
                                   months: int = 12):
        """
        Simulate network economic dynamics
        
        Args:
            num_providers (int): Number of network providers
            months (int): Simulation duration in months
        
        Returns:
            pd.DataFrame: Network economic simulation results
        """
        # Initialize simulation data
        simulation_data = []
        
        for month in range(months):
            # Simulate provider dynamics
            total_compute_power = np.random.uniform(100, 1000, num_providers)
            tasks_processed = np.random.randint(10, 100, num_providers)
            
            # Calculate network metrics
            network_compute_power = np.sum(total_compute_power)
            network_tasks = np.sum(tasks_processed)
            network_utilization = network_tasks / (num_providers * 100)
            
            # Update network metrics
            self.network_metrics.update({
                'total_providers': num_providers,
                'total_compute_power': network_compute_power,
                'total_tasks_processed': network_tasks,
                'network_utilization': network_utilization
            })
            
            # Calculate token price based on network metrics
            token_price_change = (
                0.01 * network_utilization + 
                np.random.normal(0, 0.02)
            )
            new_token_price = self.token_price_history[-1] * (1 + token_price_change)
            self.token_price_history.append(new_token_price)
            
            # Calculate provider rewards
            monthly_rewards = [
                self.calculate_provider_rewards(
                    compute_power, 
                    tasks, 
                    network_utilization
                )
                for compute_power, tasks in zip(total_compute_power, tasks_processed)
            ]
            
            # Store simulation data
            simulation_data.append({
                'month': month,
                'num_providers': num_providers,
                'total_compute_power': network_compute_power,
                'total_tasks': network_tasks,
                'network_utilization': network_utilization,
                'token_price': new_token_price,
                'total_provider_rewards': np.sum(monthly_rewards)
            })
        
        return pd.DataFrame(simulation_data)

    def visualize_token_economics(self, simulation_data: pd.DataFrame):
        """
        Visualize token economic simulation results
        
        Args:
            simulation_data (pd.DataFrame): Simulation results
        """
        plt.figure(figsize=(15, 10))
        
        # Token Price
        plt.subplot(2, 2, 1)
        plt.plot(simulation_data['month'], simulation_data['token_price'])
        plt.title('Token Price Dynamics')
        plt.xlabel('Month')
        plt.ylabel('Token Price')
        
        # Network Utilization
        plt.subplot(2, 2, 2)
        plt.plot(simulation_data['month'], simulation_data['network_utilization'])
        plt.title('Network Utilization')
        plt.xlabel('Month')
        plt.ylabel('Utilization Ratio')
        
        # Total Provider Rewards
        plt.subplot(2, 2, 3)
        plt.plot(simulation_data['month'], simulation_data['total_provider_rewards'])
        plt.title('Total Provider Rewards')
        plt.xlabel('Month')
        plt.ylabel('Tokens Rewarded')
        
        # Total Compute Power
        plt.subplot(2, 2, 4)
        plt.plot(simulation_data['month'], simulation_data['total_compute_power'])
        plt.title('Total Network Compute Power')
        plt.xlabel('Month')
        plt.ylabel('Compute Power')
        
        plt.tight_layout()
        plt.show()

def main():
    # Demonstrate token economic model
    token_model = DAICNTokenEconomics()
    
    # Simulate token distribution
    token_model.simulate_token_distribution()
    
    # Run network economics simulation
    simulation_results = token_model.simulate_network_economics(
        num_providers=100, 
        months=12
    )
    
    # Visualize results
    token_model.visualize_token_economics(simulation_results)
    
    # Print final network metrics
    print("\nFinal Network Metrics:")
    for metric, value in token_model.network_metrics.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    main()
