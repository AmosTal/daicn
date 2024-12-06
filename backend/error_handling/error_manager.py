import asyncio
import traceback
import logging
from enum import Enum, auto
from typing import Dict, Any, Optional, Callable, Coroutine

from ..logging.minimal_logger import MinimalLogger
from ..config.minimal_config import MinimalConfigurationManager

class ErrorSeverity(Enum):
    """
    Categorization of error severity levels
    """
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class ErrorCategory(Enum):
    """
    Categorization of error types
    """
    AUTHENTICATION = auto()
    RESOURCE_ALLOCATION = auto()
    TASK_PROCESSING = auto()
    NETWORK = auto()
    SYSTEM = auto()

class DAICNError(Exception):
    """
    Base custom exception for DAICN system
    """
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        self.message = message
        self.category = category
        self.severity = severity
        super().__init__(self.message)

class ErrorManager:
    """
    Centralized error management and recovery system
    
    Key Design Principles:
    - Comprehensive error tracking
    - Automatic error classification
    - Configurable recovery strategies
    - Minimal performance overhead
    """
    
    def __init__(
        self, 
        config: Optional[MinimalConfigurationManager] = None,
        logger: Optional[MinimalLogger] = None
    ):
        """
        Initialize error management system
        
        Args:
            config (Optional[MinimalConfigurationManager]): Configuration manager
            logger (Optional[MinimalLogger]): Logging system
        """
        self.config = config or MinimalConfigurationManager()
        self.logger = logger or MinimalLogger()
        
        # Error tracking
        self.error_count: Dict[ErrorCategory, int] = {
            category: 0 for category in ErrorCategory
        }
        
        # Recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
    
    def register_recovery_strategy(
        self, 
        category: ErrorCategory, 
        strategy: Callable
    ):
        """
        Register a recovery strategy for a specific error category
        
        Args:
            category (ErrorCategory): Error category to handle
            strategy (Callable): Recovery strategy function
        """
        self.recovery_strategies[category] = strategy
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Centralized error handling and recovery method
        
        Args:
            error (Exception): Caught exception
            context (Optional[Dict[str, Any]]): Additional error context
        
        Returns:
            Dict[str, Any]: Error handling result
        """
        # Classify error
        category = self._classify_error(error)
        severity = self._determine_severity(error)
        
        # Log error
        self.logger.error(
            "System error encountered", 
            {
                "error_message": str(error),
                "category": category.name,
                "severity": severity.name,
                "traceback": traceback.format_exc(),
                **(context or {})
            }
        )
        
        # Increment error count
        self.error_count[category] += 1
        
        # Attempt recovery if strategy exists
        recovery_result = await self._attempt_recovery(
            error, category, severity
        )
        
        return {
            "handled": True,
            "category": category.name,
            "severity": severity.name,
            "recovery_attempted": recovery_result['success'],
            "recovery_details": recovery_result
        }
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """
        Classify error based on exception type
        
        Args:
            error (Exception): Exception to classify
        
        Returns:
            ErrorCategory: Classified error category
        """
        error_type_mapping = {
            DAICNError: ErrorCategory.SYSTEM,
            PermissionError: ErrorCategory.AUTHENTICATION,
            MemoryError: ErrorCategory.RESOURCE_ALLOCATION,
            ConnectionError: ErrorCategory.NETWORK,
            RuntimeError: ErrorCategory.TASK_PROCESSING
        }
        
        return next(
            (
                category 
                for error_type, category in error_type_mapping.items() 
                if isinstance(error, error_type)
            ), 
            ErrorCategory.SYSTEM
        )
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """
        Determine error severity
        
        Args:
            error (Exception): Exception to assess
        
        Returns:
            ErrorSeverity: Determined severity level
        """
        severity_rules = [
            (lambda e: isinstance(e, MemoryError), ErrorSeverity.CRITICAL),
            (lambda e: isinstance(e, PermissionError), ErrorSeverity.HIGH),
            (lambda e: isinstance(e, ConnectionError), ErrorSeverity.MEDIUM),
            (lambda e: isinstance(e, RuntimeError), ErrorSeverity.LOW)
        ]
        
        return next(
            (severity for predicate, severity in severity_rules if predicate(error)), 
            ErrorSeverity.LOW
        )
    
    async def _attempt_recovery(
        self, 
        error: Exception, 
        category: ErrorCategory, 
        severity: ErrorSeverity
    ) -> Dict[str, Any]:
        """
        Attempt error recovery based on category and severity
        
        Args:
            error (Exception): Original error
            category (ErrorCategory): Error category
            severity (ErrorSeverity): Error severity
        
        Returns:
            Dict[str, Any]: Recovery attempt result
        """
        # Check if recovery strategy exists
        if category not in self.recovery_strategies:
            return {
                "success": False,
                "message": "No recovery strategy defined"
            }
        
        try:
            # Execute recovery strategy
            recovery_result = await self.recovery_strategies[category](
                error, severity
            )
            
            return {
                "success": True,
                "details": recovery_result
            }
        
        except Exception as recovery_error:
            # Log recovery failure
            self.logger.error(
                "Error recovery failed", 
                {
                    "original_error": str(error),
                    "recovery_error": str(recovery_error)
                }
            )
            
            return {
                "success": False,
                "message": "Recovery attempt failed"
            }
    
    def get_error_statistics(self) -> Dict[str, int]:
        """
        Retrieve error statistics
        
        Returns:
            Dict[str, int]: Error count by category
        """
        return {
            category.name: count 
            for category, count in self.error_count.items()
        }

async def default_recovery_strategies(
    error: Exception, 
    severity: ErrorSeverity
) -> Dict[str, Any]:
    """
    Default recovery strategies for different error scenarios
    
    Args:
        error (Exception): Error to recover from
        severity (ErrorSeverity): Error severity
    
    Returns:
        Dict[str, Any]: Recovery action details
    """
    if severity == ErrorSeverity.CRITICAL:
        # Aggressive recovery for critical errors
        return {"action": "system_restart"}
    
    elif severity == ErrorSeverity.HIGH:
        # Partial system reset
        return {"action": "component_restart"}
    
    elif severity == ErrorSeverity.MEDIUM:
        # Retry mechanism
        return {"action": "retry", "max_attempts": 3}
    
    else:
        # Minimal intervention
        return {"action": "log_and_continue"}

def main():
    """
    Demonstration of Error Management System
    """
    async def run_demo():
        error_manager = ErrorManager()
        
        # Register default recovery strategies
        error_manager.register_recovery_strategy(
            ErrorCategory.SYSTEM, 
            default_recovery_strategies
        )
        
        # Simulate various error scenarios
        try:
            raise RuntimeError("Simulated task processing error")
        except Exception as e:
            result = await error_manager.handle_error(e)
            print("Error Handling Result:", result)
        
        # Display error statistics
        print("Error Statistics:", error_manager.get_error_statistics())
    
    asyncio.run(run_demo())

if __name__ == '__main__':
    main()
