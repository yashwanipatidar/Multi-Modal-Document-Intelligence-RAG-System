# 📊 Detailed Operation Tracking - Quick Reference

## What's Been Added?

Complete operation tracking system with detailed logging for every step in the RAG system.

### 🎯 Tracked Operations

| Operation | What's Logged | Timing |
|-----------|--------------|--------|
| **PDF Upload** | Files, sizes, success status | ⏱️ Total upload duration |
| **Text & Image Extraction** | Chunks extracted, OCR status | ⏱️ Extraction time |
| **Table Extraction** | Tables found, PDF count | ⏱️ Processing time |
| **Index Building** | Chunks processed, table count, file size | ⏱️ Total build time |
| **Query Retrieval** | Results found, top-k, retrieval time | ⏱️ Search duration |
| **LLM Generation** | Query, answer length, generation time | ⏱️ LLM response time |

---

## 📍 Where to See Logs

### 1️⃣ **Streamlit Sidebar** (Main Location)
Open the app and look at the right sidebar:

```
🔧 Configuration
├── 📁 Upload PDFs
├── 📚 Document Index
├── ⚙️ Retrieval Settings
├── 📊 Document Status
│   ├── PDFs: 3
│   ├── Tables: 12
│   ├── Images: 45
│   └── Index: Ready
└── 📋 Operation Logs  ← CLICK HERE
    ├── Total Ops: 8
    ├── Success Rate: 100%
    ├── Total Time: 25.4s
    └── 📋 Show Detailed Logs  ← EXPAND THIS
        ├── ✅ PDF Upload (5.34s)
        ├── ✅ Text & Image Extraction (12.1s)
        ├── ✅ Table Extraction (2.5s)
        └── ✅ Index Building (5.5s)
```

Click any operation to see:
- ⏰ **Started:** Timestamp
- ⏱️ **Duration:** How long it took
- 🔄 **Status:** Success/Failed/Warning
- 📝 **Details:** Specific metrics
- ⚠️ **Warnings:** Any issues

### 2️⃣ **File System** (For Analysis)
Location: `data/processed/session_logs/`

```
session_logs/
├── operations_abc123xyz.jsonl  ← JSON logs (one per line)
├── debug_abc123xyz.log         ← Full debug logs
├── operations_def456uvw.jsonl
├── debug_def456uvw.log
└── ... (one pair per session)
```

---

## 🚀 How to Use

### View Logs in Streamlit
1. Run: `streamlit run src/demo_app.py`
2. Look at sidebar → **📋 Operation Logs**
3. Click "📋 Show Detailed Logs" to expand
4. Click any operation to see full details

### Run Test Examples
```bash
python test_operation_tracking.py
```

This will:
- Create sample operations
- Show timing information
- Demonstrate error handling
- Display log analysis

### Access Logs Programmatically
```python
from src.operation_tracker import get_operation_tracker
from src.config import PROCESSED_DIR

tracker = get_operation_tracker("my_session", PROCESSED_DIR / "session_logs")

# Get summary
summary = tracker.get_operations_summary()
print(f"Total operations: {summary['total_operations']}")
print(f"Success rate: {summary['success_rate']}%")
print(f"Total time: {summary['total_duration']}s")

# Get detailed logs
details = tracker.get_operation_details()
for op in details:
    print(f"{op['operation_name']}: {op['duration_seconds']}s")
```

---

## 📊 Example Operation Log

### What You See in Streamlit
```
✅ PDF Upload (5.34s)
├─ Started: 2025-05-17T14:32:10.123456
├─ Status: success
├─ Duration: 5.34s
└─ Details:
   ├─ file_count: 2
   ├─ saved_files: 2
   ├─ file_report.pdf: {size_mb: 2.5, status: saved}
   └─ file_data.pdf: {size_mb: 1.8, status: saved}
```

### What's Stored in File (JSON)
```json
{
  "operation_name": "PDF Upload",
  "status": "success",
  "start_time": "2025-05-17T14:32:10.123456",
  "end_time": "2025-05-17T14:32:15.456789",
  "duration_seconds": 5.33,
  "details": {
    "file_count": 2,
    "saved_files": 2,
    "file_report.pdf": {"size_mb": 2.5, "status": "saved"},
    "file_data.pdf": {"size_mb": 1.8, "status": "saved"}
  },
  "error_message": null,
  "warning_messages": []
}
```

---

## 🔍 Key Metrics Tracked

### PDF Upload
- ✅ Number of files
- ✅ File sizes (MB)
- ✅ File paths
- ✅ Success/failure count

### Text & Image Extraction
- ✅ PDF count
- ✅ Chunks extracted
- ✅ OCR enabled/disabled
- ✅ Images extracted

### Table Extraction
- ✅ PDF count
- ✅ Tables found
- ✅ Table paths
- ✅ Success count

### Index Building
- ✅ Chunks processed
- ✅ Tables indexed
- ✅ Index file size (MB)
- ✅ Metadata size

### Query Retrieval
- ✅ Query text (first 100 chars)
- ✅ Top-k parameter
- ✅ Results found count
- ✅ Retrieval time (seconds)

---

## ✨ Key Features

✅ **Automatic Tracking** - No code changes needed
✅ **Real-Time Display** - See logs as operations complete
✅ **Detailed Metrics** - Track what matters
✅ **Error Capture** - Full error messages and stack traces
✅ **Performance Data** - Timing for every operation
✅ **File Logging** - JSON format for analysis
✅ **Session Isolation** - Each user session has separate logs
✅ **Expandable UI** - View summary or detailed logs

---

## 🎯 Common Use Cases

### Monitor Performance
1. Go to Streamlit sidebar
2. Check "📋 Operation Logs"
3. View success rates and timing
4. Identify slow operations

### Debug Failed Operations
1. Find ❌ Failed operation in logs
2. Expand to see error message
3. Check warning messages
4. Check full debug log file for stack trace

### Analyze System Health
```python
tracker.get_operations_summary()
# Returns: {
#   "total_operations": 25,
#   "success_rate": 96.0,
#   "total_duration": 125.5,
#   "success_count": 24,
#   "failed_count": 1
# }
```

---

## 📁 Files Added/Modified

### New Files
- `src/operation_tracker.py` - Core tracking module
- `test_operation_tracking.py` - Test examples
- `OPERATION_TRACKING_GUIDE.txt` - Detailed documentation
- `OPERATION_TRACKING_QUICK_REFERENCE.md` - This file

### Modified Files
- `src/demo_app.py` - Added tracking to all operations
  - Import tracker module
  - Initialize tracker per session
  - Wrap key operations with timing
  - Add sidebar log display

---

## 🔄 Session Workflow

```
User Uploads PDFs
        ↓
   [TRACKED] ✅ PDF Upload (5.3s)
        ↓
User Clicks "Build Index"
        ↓
   [TRACKED] ✅ Text & Image Extraction (12.1s)
        ↓
   [TRACKED] ✅ Table Extraction (2.5s)
        ↓
   [TRACKED] ✅ Index Building (5.5s)
        ↓
User Asks Question
        ↓
   [TRACKED] ✅ Query Retrieval (1.2s)
        ↓
   [TRACKED] ✅ Answer Generated (2.1s)
        ↓
Results Displayed + Logs Updated
```

---

## 🚀 Next Steps

1. **Run the app:**
   ```bash
   streamlit run src/demo_app.py
   ```

2. **Upload a PDF and watch logs update:**
   - Check sidebar "📋 Operation Logs"
   - See metrics in real-time

3. **Run test examples:**
   ```bash
   python test_operation_tracking.py
   ```

4. **Analyze logs programmatically:**
   - Read JSON files from `session_logs/`
   - Create performance reports
   - Monitor system health

---

## 📚 For More Details
See: `OPERATION_TRACKING_GUIDE.txt`

---

**Last Updated:** May 17, 2025
**System Status:** ✅ Production Ready
