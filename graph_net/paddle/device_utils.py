import paddle
import json
import platform
from typing import Dict, Any, Optional


class DeviceConfig:
    """设备配置管理器"""
    
    # 预定义的设备配置模板
    DEVICE_CONFIGS = {
        "cuda": {
            "backend": "cuda",
            "memory_allocator": "default",
            "compute_capability": None,  # 自动检测
            "compiler_flags": ["-O3"]
        },
        "dcu": {
            "backend": "rocm",
            "memory_allocator": "cnmem",
            "compute_units": 64,
            "compiler_flags": ["-O3", "-fopenmp"]
        },
        "cpu": {
            "backend": "cpu",
            "memory_allocator": "default",
            "threads": "auto",
            "compiler_flags": ["-O2"]
        }
    }
    
    # 编译器后端配置
    COMPILER_BACKENDS = {
        "cinn": {
            "module": "paddle.jit",
            "config": {
                "build_strategy": "DEFAULT",
                "full_graph": True
            }
        },
        "nope": {
            "module": None,
            "config": {}  # 不使用编译器，直接运行
        }
    }
    
    @classmethod
    def get_device_config(cls, device: str) -> Dict[str, Any]:
        """获取设备配置"""
        config = cls.DEVICE_CONFIGS.get(device.lower(), {}).copy()
        
        # 自动检测设备能力
        if device.lower() == "cuda":
            if paddle.device.cuda.device_count() > 0:
                config["compute_capability"] = paddle.device.cuda.get_device_capability(0)
        
        return config
    
    @classmethod
    def get_backend_config(cls, backend: str) -> Optional[Dict[str, Any]]:
        """获取编译器后端配置"""
        return cls.COMPILER_BACKENDS.get(backend)
    
    @classmethod
    def get_hardware_info(cls, device: str) -> Dict[str, Any]:
        """获取硬件信息"""
        info = {
            "device_type": device,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "paddle_version": paddle.__version__
        }
        
        if device.lower() == "cuda":
            if paddle.device.cuda.device_count() > 0:
                info.update({
                    "device_name": paddle.device.cuda.get_device_name(0),
                    "device_count": paddle.device.cuda.device_count(),
                    "compute_capability": paddle.device.cuda.get_device_capability(0),
                    "memory_total": paddle.device.cuda.get_device_properties(0).total_memory
                })
        
        return info


class DeviceEnvironment:
    """设备环境管理器"""
    
    def __init__(self, device: str, backend: str = "cinn"):
        self.device = device
        self.backend = backend
        self.device_config = DeviceConfig.get_device_config(device)
        self.backend_config = DeviceConfig.get_backend_config(backend)
        self.hardware_info = DeviceConfig.get_hardware_info(device)
    
    def setup_environment(self):
        """设置运行环境"""
        # 设置设备
        if self.device.lower() == "cuda":
            paddle.set_device('gpu:0')
        elif self.device.lower() == "dcu":
            paddle.set_device('gpu:0')  # DCU也使用gpu设备
        else:
            paddle.set_device('cpu')
        
        # 设置线程数（CPU模式）
        if self.device.lower() == "cpu":
            import os
            # 自动设置为可用CPU核心数
            cpu_count = os.cpu_count()
            if cpu_count:
                paddle.set_num_threads(cpu_count)
    
    def validate_environment(self) -> bool:
        """验证环境是否可用"""
        try:
            if self.device.lower() == "cuda":
                return paddle.device.cuda.device_count() > 0
            elif self.device.lower() == "dcu":
                # DCU检测逻辑
                return True  # 简化处理
            return True
        except Exception:
            return False
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """获取环境摘要"""
        return {
            "hardware": self.hardware_info,
            "device_config": self.device_config,
            "backend_config": self.backend_config,
            "environment_valid": self.validate_environment()
        }


def create_device_spec_file(result_dir: str, device_env: DeviceEnvironment):
    """创建设备规格文件"""
    spec_data = device_env.get_environment_summary()
    
    spec_file = f"{result_dir}/device_spec.json"
    with open(spec_file, 'w') as f:
        json.dump(spec_data, f, indent=2)
    
    return spec_file