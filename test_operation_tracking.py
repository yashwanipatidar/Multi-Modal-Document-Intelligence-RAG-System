#!/usr/bin/env python3
"""
Test and demonstration script for Operation Tracking System

This script shows:
1. How to use the operation tracker
2. How to access logs
3. How to analyze operation metrics
4. Error handling examples
"""

import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.operation_tracker import (
    get_operation_tracker, 
    OperationStatus, 
    OperationTimer
)
from src.config import PROCESSED_DIR


def example_1_basic_tracking():
    """Example 1: Basic operation tracking"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Operation Tracking")
    print("="*60)
    
    tracker = get_operation_tracker("test_session_1", log_dir=PROCESSED_DIR / "session_logs")
    
    # Track a simple operation
    with OperationTimer(tracker, "Download Data", {"source": "API", "items": 100}):
        time.sleep(0.5)  # Simulate work
        tracker.add_detail("bytes_downloaded", 1024 * 500)  # 500KB
    
    # Get summary
    summary = tracker.get_operations_summary()
    print(f"\n✅ Operation Summary:")
    print(f"   Total Operations: {summary['total_operations']}")
    print(f"   Success Rate: {summary['success_rate']:.0f}%")
    print(f"   Total Duration: {summary['total_duration']:.2f}s")


def example_2_error_handling():
    """Example 2: Error handling and failed operations"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Error Handling")
    print("="*60)
    
    tracker = get_operation_tracker("test_session_2", log_dir=PROCESSED_DIR / "session_logs")
    
    # Track an operation that fails
    try:
        with OperationTimer(tracker, "Process Data", {"rows": 1000}):
            time.sleep(0.3)
            raise ValueError("Invalid data format")
    except ValueError as e:
        print(f"\n❌ Operation Failed: {e}")
    
    # Track a successful operation
    with OperationTimer(tracker, "Validate Schema", {"tables": 5}):
        time.sleep(0.2)
        tracker.add_detail("valid_tables", 5)
    
    # Get summary
    summary = tracker.get_operations_summary()
    print(f"\n✅ Operation Summary:")
    print(f"   Total Operations: {summary['total_operations']}")
    print(f"   Successful: {summary['success_count']}")
    print(f"   Failed: {summary['failed_count']}")
    print(f"   Success Rate: {summary['success_rate']:.0f}%")


def example_3_detailed_metrics():
    """Example 3: Adding detailed metrics"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Detailed Metrics")
    print("="*60)
    
    tracker = get_operation_tracker("test_session_3", log_dir=PROCESSED_DIR / "session_logs")
    
    # Multi-step operation with detailed tracking
    with OperationTimer(tracker, "Index Building", {"mode": "multi-modal"}):
        # Step 1: Extract
        tracker.add_detail("step_1_extract", {"files": 5, "status": "started"})
        time.sleep(0.2)
        tracker.add_detail("step_1_extract", {"files": 5, "status": "completed", "chunks": 250})
        
        # Step 2: Embed
        tracker.add_detail("step_2_embed", {"vectors": 250, "status": "started"})
        time.sleep(0.3)
        tracker.add_detail("step_2_embed", {"vectors": 250, "status": "completed"})
        
        # Step 3: Index
        tracker.add_detail("step_3_index", {"status": "building"})
        time.sleep(0.2)
        tracker.add_detail("step_3_index", {"status": "completed", "size_mb": 45.2})
    
    # Display details
    details = tracker.get_operation_details("Index Building")
    print(f"\n✅ Index Building Details:")
    for op in details:
        print(f"\n   Operation: {op['operation_name']}")
        print(f"   Duration: {op['duration_seconds']:.2f}s")
        print(f"   Details: {json.dumps(op['details'], indent=6)}")


def example_4_warnings():
    """Example 4: Operations with warnings"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Operations with Warnings")
    print("="*60)
    
    tracker = get_operation_tracker("test_session_4", log_dir=PROCESSED_DIR / "session_logs")
    
    with OperationTimer(tracker, "Data Cleaning", {"rows": 10000}):
        tracker.add_detail("processed_rows", 9800)
        tracker.add_warning("Skipped 200 malformed rows")
        tracker.add_warning("Memory usage at 85%")
    
    # Get details with warnings
    details = tracker.get_operation_details("Data Cleaning")
    print(f"\n✅ Data Cleaning Operation:")
    for op in details:
        print(f"   Status: {op['status']}")
        print(f"   Details: {op['details']}")
        if op['warning_messages']:
            print(f"   Warnings: {op['warning_messages']}")


def example_5_log_analysis():
    """Example 5: Analyzing logs from file"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Log Analysis")
    print("="*60)
    
    # Create some operations
    tracker = get_operation_tracker("test_session_5", log_dir=PROCESSED_DIR / "session_logs")
    
    operations = [
        ("Query Processing", 1.5),
        ("Retrieval", 2.3),
        ("Answer Generation", 3.1),
    ]
    
    for op_name, duration in operations:
        with OperationTimer(tracker, op_name, {}):
            time.sleep(duration / 10)  # Scaled down for demo
    
    # Read logs from file
    log_file = PROCESSED_DIR / "session_logs" / f"operations_test_session_5.jsonl"
    
    if log_file.exists():
        print(f"\n✅ Reading logs from: {log_file}")
        
        total_time = 0
        operation_count = 0
        
        with open(log_file) as f:
            for line in f:
                op = json.loads(line)
                total_time += op['duration_seconds']
                operation_count += 1
                print(f"\n   {op['operation_name']}:")
                print(f"      Duration: {op['duration_seconds']:.2f}s")
                print(f"      Status: {op['status'].upper()}")
        
        print(f"\n✅ Summary:")
        print(f"   Total Operations: {operation_count}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Time: {total_time/operation_count:.2f}s")
    else:
        print(f"   ⚠️  Log file not found: {log_file}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "OPERATION TRACKING SYSTEM - EXAMPLES" + " "*13 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        example_1_basic_tracking()
        example_2_error_handling()
        example_3_detailed_metrics()
        example_4_warnings()
        example_5_log_analysis()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\n📚 Next steps:")
        print("   1. Check data/processed/session_logs/ for log files")
        print("   2. Run 'streamlit run src/demo_app.py' to see live tracking")
        print("   3. Read OPERATION_TRACKING_GUIDE.txt for detailed docs")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
