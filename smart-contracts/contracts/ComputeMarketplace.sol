// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "./DAICNToken.sol";

contract ComputeMarketplace is ReentrancyGuard, Ownable {
    DAICNToken public token;
    
    struct Provider {
        address addr;
        uint256 computePower;  // in FLOPS
        uint256 reputation;
        bool isActive;
        uint256 pricePerUnit;  // price in DAICN tokens per compute unit
    }
    
    struct Task {
        address client;
        uint256 computeUnits;
        uint256 reward;
        address provider;
        bool completed;
        bytes32 resultHash;
    }
    
    mapping(address => Provider) public providers;
    mapping(uint256 => Task) public tasks;
    uint256 public taskCount;
    
    event ProviderRegistered(address indexed provider, uint256 computePower);
    event TaskCreated(uint256 indexed taskId, address indexed client, uint256 computeUnits);
    event TaskCompleted(uint256 indexed taskId, address indexed provider, bytes32 resultHash);
    
    constructor(address _tokenAddress) {
        token = DAICNToken(_tokenAddress);
    }
    
    function registerProvider(uint256 _computePower, uint256 _pricePerUnit) external {
        require(_computePower > 0, "Invalid compute power");
        require(_pricePerUnit > 0, "Invalid price");
        
        providers[msg.sender] = Provider({
            addr: msg.sender,
            computePower: _computePower,
            reputation: 100,
            isActive: true,
            pricePerUnit: _pricePerUnit
        });
        
        emit ProviderRegistered(msg.sender, _computePower);
    }
    
    function createTask(uint256 _computeUnits) external payable nonReentrant {
        require(_computeUnits > 0, "Invalid compute units");
        
        uint256 taskId = taskCount++;
        tasks[taskId] = Task({
            client: msg.sender,
            computeUnits: _computeUnits,
            reward: msg.value,
            provider: address(0),
            completed: false,
            resultHash: bytes32(0)
        });
        
        emit TaskCreated(taskId, msg.sender, _computeUnits);
    }
    
    function completeTask(uint256 _taskId, bytes32 _resultHash) external nonReentrant {
        Task storage task = tasks[_taskId];
        require(!task.completed, "Task already completed");
        require(task.provider == msg.sender, "Not assigned provider");
        
        task.completed = true;
        task.resultHash = _resultHash;
        
        // Transfer reward
        token.transfer(msg.sender, task.reward);
        
        emit TaskCompleted(_taskId, msg.sender, _resultHash);
    }
    
    // Additional functions for task assignment, reputation updates, etc.
}
