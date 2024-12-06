// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ProviderRegistry {
    struct Provider {
        address providerAddress;
        uint256 computeCapacity;  // In compute units
        uint256 reputation;
        bool isRegistered;
        uint256 stakedTokens;
    }

    mapping(address => Provider) public providers;
    address[] public registeredProviders;

    // Token contract interface
    IERC20 public token;

    // Minimum stake required to register
    uint256 public constant MINIMUM_STAKE = 1000 * 10**18;  // 1000 tokens

    event ProviderRegistered(address indexed provider, uint256 computeCapacity);
    event ProviderUpdated(address indexed provider, uint256 newComputeCapacity);
    event ProviderSlashed(address indexed provider, uint256 slashedAmount);

    constructor(address _tokenAddress) {
        token = IERC20(_tokenAddress);
    }

    function registerProvider(uint256 computeCapacity) external {
        require(!providers[msg.sender].isRegistered, "Provider already registered");
        require(token.transferFrom(msg.sender, address(this), MINIMUM_STAKE), "Stake transfer failed");

        providers[msg.sender] = Provider({
            providerAddress: msg.sender,
            computeCapacity: computeCapacity,
            reputation: 100,  // Start with perfect reputation
            isRegistered: true,
            stakedTokens: MINIMUM_STAKE
        });

        registeredProviders.push(msg.sender);

        emit ProviderRegistered(msg.sender, computeCapacity);
    }

    function updateComputeCapacity(uint256 newCapacity) external {
        require(providers[msg.sender].isRegistered, "Provider not registered");
        providers[msg.sender].computeCapacity = newCapacity;
        emit ProviderUpdated(msg.sender, newCapacity);
    }

    function slashProvider(address providerAddress, uint256 slashAmount) external {
        require(providers[providerAddress].isRegistered, "Provider not registered");
        require(slashAmount <= providers[providerAddress].stakedTokens, "Slash amount exceeds staked tokens");

        providers[providerAddress].stakedTokens -= slashAmount;
        providers[providerAddress].reputation -= 10;  // Reduce reputation

        // Transfer slashed tokens to a designated address (could be a treasury or reward pool)
        token.transfer(address(this), slashAmount);

        emit ProviderSlashed(providerAddress, slashAmount);
    }

    function getProviderCount() external view returns (uint256) {
        return registeredProviders.length;
    }

    function getProviderDetails(address providerAddress) external view returns (Provider memory) {
        return providers[providerAddress];
    }
}

// Minimal ERC20 interface for token interactions
interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
}
