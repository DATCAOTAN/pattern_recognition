"""
System Monitor Module - GPU, RAM, CPU Monitoring
"""
import psutil
from typing import Dict, Any
import platform

try:
    import GPUtil
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.model_status = "Not Loaded"
        self.model_path = None
        
    def set_model_status(self, status: str, path: str = None):
        """Update model status"""
        self.model_status = status
        self.model_path = path
        
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_ram_usage(self) -> Dict[str, Any]:
        """Get RAM usage information"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent
        }
    
    def get_gpu_usage(self) -> Dict[str, Any]:
        """Get GPU usage information"""
        if not GPU_AVAILABLE:
            return {
                "available": False,
                "message": "No GPU detected or GPUtil not installed"
            }
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {
                    "available": False,
                    "message": "No GPU detected"
                }
            
            gpu = gpus[0]  # Use first GPU
            return {
                "available": True,
                "name": gpu.name,
                "memory_total_mb": gpu.memoryTotal,
                "memory_used_mb": gpu.memoryUsed,
                "memory_free_mb": gpu.memoryFree,
                "memory_percent": round(gpu.memoryUsed / gpu.memoryTotal * 100, 1) if gpu.memoryTotal > 0 else 0,
                "gpu_load_percent": round(gpu.load * 100, 1),
                "temperature": gpu.temperature
            }
        except Exception as e:
            return {
                "available": False,
                "message": f"Error getting GPU info: {str(e)}"
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get overall system information"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count()
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        gpu_info = self.get_gpu_usage()
        ram_info = self.get_ram_usage()
        
        # Determine overall health status
        health = "Optimal"
        if ram_info["percent"] > 90:
            health = "Critical"
        elif ram_info["percent"] > 75:
            health = "Warning"
        
        if gpu_info.get("available") and gpu_info.get("memory_percent", 0) > 90:
            health = "Critical" if health != "Critical" else health
        
        return {
            "health": health,
            "model_status": self.model_status,
            "model_path": str(self.model_path) if self.model_path else None,
            "cpu_percent": self.get_cpu_usage(),
            "ram": ram_info,
            "gpu": gpu_info,
            "system_info": self.get_system_info()
        }


# Singleton instance
system_monitor = SystemMonitor()
