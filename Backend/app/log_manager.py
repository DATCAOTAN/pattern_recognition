"""
Log Manager Module - Activity Logging and Export
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from threading import Lock

from .config import LOGS_DIR


class LogManager:
    """Manages detection logs and exports"""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lock = Lock()
        self._ensure_dirs()
        
    def _ensure_dirs(self):
        """Ensure log directories exist"""
        LOGS_DIR.mkdir(exist_ok=True)
        
    def add_log(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a detection log entry
        
        Args:
            detection: Detection data
            
        Returns:
            Log entry with timestamp
        """
        log_entry = {
            "id": len(self.logs) + 1,
            "timestamp": datetime.now().isoformat(),
            "formatted_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "class_name": detection.get("class_name", "Unknown"),
            "category": detection.get("category", "Unknown"),
            "confidence": detection.get("confidence", 0),
            "bbox": detection.get("bbox", {}),
            "session_id": self.session_id
        }
        
        with self.lock:
            self.logs.append(log_entry)
        
        return log_entry
    
    def add_batch_logs(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add multiple detection logs at once
        
        Args:
            detections: List of detection data
            
        Returns:
            List of log entries
        """
        log_entries = []
        for det in detections:
            entry = self.add_log(det)
            log_entries.append(entry)
        return log_entries
    
    def get_logs(
        self, 
        limit: Optional[int] = None, 
        class_filter: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get logs with optional filtering
        
        Args:
            limit: Maximum number of logs to return
            class_filter: Filter by class names
            category_filter: Filter by category (Organic/Inorganic)
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            
        Returns:
            Filtered list of logs
        """
        filtered = self.logs.copy()
        
        # Apply class filter
        if class_filter:
            filtered = [l for l in filtered if l["class_name"] in class_filter]
            
        # Apply category filter
        if category_filter:
            filtered = [l for l in filtered if l["category"] == category_filter]
            
        # Apply time filters
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            filtered = [l for l in filtered if datetime.fromisoformat(l["timestamp"]) >= start_dt]
            
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
            filtered = [l for l in filtered if datetime.fromisoformat(l["timestamp"]) <= end_dt]
        
        # Sort by timestamp descending
        filtered.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
            
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.logs:
            return {
                "total_detections": 0,
                "inorganic_count": 0,
                "organic_count": 0,
                "class_counts": {},
                "inorganic_percentage": 0,
                "organic_percentage": 0
            }
        
        inorganic_count = sum(1 for l in self.logs if l["category"] == "Inorganic")
        organic_count = sum(1 for l in self.logs if l["category"] == "Organic")
        total = len(self.logs)
        
        # Count by class
        class_counts = {}
        for log in self.logs:
            cls = log["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        return {
            "total_detections": total,
            "inorganic_count": inorganic_count,
            "organic_count": organic_count,
            "class_counts": class_counts,
            "inorganic_percentage": round(inorganic_count / total * 100, 1) if total > 0 else 0,
            "organic_percentage": round(organic_count / total * 100, 1) if total > 0 else 0,
            "session_id": self.session_id
        }
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export logs to CSV file
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"logs_{self.session_id}.csv"
        
        filepath = LOGS_DIR / filename
        
        if not self.logs:
            # Create empty file with headers
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Timestamp", "Class", "Category", "Confidence"])
            return str(filepath)
        
        df = pd.DataFrame(self.logs)
        df = df[["id", "formatted_time", "class_name", "category", "confidence"]]
        df.columns = ["ID", "Timestamp", "Class", "Category", "Confidence"]
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return str(filepath)
    
    def export_to_excel(self, filename: Optional[str] = None) -> str:
        """
        Export logs to Excel file
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"logs_{self.session_id}.xlsx"
        
        filepath = LOGS_DIR / filename
        
        if not self.logs:
            # Create empty DataFrame with headers
            df = pd.DataFrame(columns=["ID", "Timestamp", "Class", "Category", "Confidence"])
        else:
            df = pd.DataFrame(self.logs)
            df = df[["id", "formatted_time", "class_name", "category", "confidence"]]
            df.columns = ["ID", "Timestamp", "Class", "Category", "Confidence"]
        
        # Add statistics sheet
        stats = self.get_statistics()
        stats_df = pd.DataFrame([{
            "Total Detections": stats["total_detections"],
            "Inorganic Count": stats["inorganic_count"],
            "Organic Count": stats["organic_count"],
            "Inorganic %": stats["inorganic_percentage"],
            "Organic %": stats["organic_percentage"]
        }])
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detection Logs', index=False)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        return str(filepath)
    
    def clear_logs(self):
        """Clear all logs and start new session"""
        with self.lock:
            self.logs = []
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_session(self):
        """Save current session logs to file"""
        if self.logs:
            self.export_to_csv()


# Singleton instance
log_manager = LogManager()
