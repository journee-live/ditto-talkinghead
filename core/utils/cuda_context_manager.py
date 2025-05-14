import threading
import os
from loguru import logger

class CUDAContextManager:
    """
    Singleton class to manage CUDA context initialization.
    Ensures CUDA is initialized only once across multiple threads.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CUDAContextManager, cls).__new__(cls)
            return cls._instance

    def initialize(self):
        """Initialize CUDA context if not already initialized."""
        with self._lock:
            if not self._initialized:
                try:
                    # Set environment variable to prevent CUDA from auto-initializing
                    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
                    
                    # Import CUDA modules here to ensure controlled initialization
                    from cuda import cuda, cudart
                    
                    # Explicitly initialize CUDA driver
                    cuda.cuInit(0)
                    logger.info("CUDA context initialized successfully")
                    self._initialized = True
                except Exception as e:
                    logger.error(f"Failed to initialize CUDA context: {e}")
                    raise RuntimeError(f"CUDA initialization failed: {e}")
            return self._initialized

    def is_initialized(self):
        """Check if CUDA has been initialized."""
        return self._initialized
