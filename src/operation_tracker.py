"""
Operation Tracking and Logging Module
Provides detailed tracking of all RAG system operations with timing and status monitoring
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, asdict, field
from enum import Enum
import traceback


class OperationStatus(Enum):
    """Status of an operation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class OperationLog:
    """Single operation log entry"""
    operation_name: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class OperationTracker:
    """Track and log detailed operations with timing and status"""
    
    def __init__(self, session_id: str, log_dir: Optional[Path] = None):
        """
        Initialize operation tracker
        
        Args:
            session_id: Unique session identifier
            log_dir: Directory to save operation logs (optional)
        """
        self.session_id = session_id
        self.log_dir = log_dir
        self.operations: List[OperationLog] = []
        self.current_operation: Optional[Dict[str, Any]] = None
        
        # Setup file logging if directory provided
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = log_dir / f"operations_{session_id}.jsonl"
        else:
            self.log_file = None
            
        # Setup Python logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger"""
        logger = logging.getLogger(f"rag_tracker_{self.session_id}")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if log_dir provided
        if self.log_dir:
            file_handler = logging.FileHandler(
                self.log_dir / f"debug_{self.session_id}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def start_operation(self, operation_name: str, details: Optional[Dict[str, Any]] = None):
        """
        Start tracking an operation
        
        Args:
            operation_name: Name of the operation
            details: Optional initial details
        """
        self.current_operation = {
            "operation_name": operation_name,
            "status": OperationStatus.IN_PROGRESS.value,
            "start_time": datetime.now(),
            "details": details or {},
            "warnings": [],
        }
        self.logger.info(f"▶️  Starting: {operation_name}")
    
    def add_detail(self, key: str, value: Any):
        """Add detail to current operation"""
        if self.current_operation:
            self.current_operation["details"][key] = value
    
    def add_warning(self, message: str):
        """Add warning to current operation"""
        if self.current_operation:
            self.current_operation["warnings"].append(message)
            self.logger.warning(f"⚠️  {message}")
    
    def end_operation(self, status: OperationStatus = OperationStatus.SUCCESS, 
                      error: Optional[Exception] = None):
        """
        End tracking current operation
        
        Args:
            status: Operation status
            error: Optional exception if failed
        """
        if not self.current_operation:
            return
        
        end_time = datetime.now()
        duration = (end_time - self.current_operation["start_time"]).total_seconds()
        
        error_message = None
        if error:
            error_message = f"{type(error).__name__}: {str(error)}"
            self.logger.error(f"❌ Error: {error_message}")
        
        # Create log entry
        log_entry = OperationLog(
            operation_name=self.current_operation["operation_name"],
            status=status.value,
            start_time=self.current_operation["start_time"].isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            details=self.current_operation["details"],
            error_message=error_message,
            warning_messages=self.current_operation["warnings"],
        )
        
        self.operations.append(log_entry)
        
        # Log to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry.to_dict()) + "\n")
        
        # Console summary
        status_symbol = {
            OperationStatus.SUCCESS: "✅",
            OperationStatus.FAILED: "❌",
            OperationStatus.WARNING: "⚠️",
            OperationStatus.PENDING: "⏳",
        }.get(status, "ℹ️")
        
        self.logger.info(
            f"{status_symbol} Completed: {self.current_operation['operation_name']} "
            f"({duration:.2f}s)"
        )
        
        self.current_operation = None
    
    def get_operations_summary(self) -> Dict[str, Any]:
        """Get summary of all operations"""
        if not self.operations:
            return {
                "total_operations": 0,
                "total_duration": 0,
                "success_count": 0,
                "failed_count": 0,
                "warning_count": 0,
            }
        
        total_duration = sum(op.duration_seconds for op in self.operations)
        success_count = len([op for op in self.operations if op.status == OperationStatus.SUCCESS.value])
        failed_count = len([op for op in self.operations if op.status == OperationStatus.FAILED.value])
        warning_count = len([op for op in self.operations if op.status == OperationStatus.WARNING.value])
        
        return {
            "total_operations": len(self.operations),
            "total_duration": total_duration,
            "success_count": success_count,
            "failed_count": failed_count,
            "warning_count": warning_count,
            "success_rate": (success_count / len(self.operations) * 100) if self.operations else 0,
        }
    
    def get_operation_details(self, operation_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed logs for operations"""
        ops = self.operations
        if operation_name:
            ops = [op for op in ops if op.operation_name == operation_name]
        
        return [op.to_dict() for op in ops]


class OperationTimer:
    """Context manager for timing operations"""
    
    def __init__(self, tracker: OperationTracker, operation_name: str, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize timer
        
        Args:
            tracker: OperationTracker instance
            operation_name: Name of operation
            details: Optional initial details
        """
        self.tracker = tracker
        self.operation_name = operation_name
        self.details = details
    
    def __enter__(self):
        """Start operation"""
        self.tracker.start_operation(self.operation_name, self.details)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End operation"""
        if exc_type is not None:
            # Operation failed
            self.tracker.end_operation(
                status=OperationStatus.FAILED,
                error=exc_val
            )
            return False  # Re-raise the exception
        else:
            # Operation succeeded
            self.tracker.end_operation(status=OperationStatus.SUCCESS)
            return True


def get_operation_tracker(session_id: str, log_dir: Optional[Path] = None) -> OperationTracker:
    """
    Factory function to get or create operation tracker
    
    Args:
        session_id: Unique session identifier
        log_dir: Optional directory for logs
    
    Returns:
        OperationTracker instance
    """
    return OperationTracker(session_id, log_dir)
