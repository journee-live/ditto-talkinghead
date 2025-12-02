import ctypes
import os
from collections import OrderedDict

import numpy as np
from cuda import cuda, cudart, nvrtc

from ..utils.cuda_context_manager import CUDAContextManager

try:
    import tensorrt as trt
except ImportError:
    import tensorrt_libs

    trt_libs_path = tensorrt_libs.__path__[0]
    ctypes.CDLL(os.path.join(trt_libs_path, "libnvinfer.so.8"))
    ctypes.CDLL(os.path.join(trt_libs_path, "libnvinfer_plugin.so.8"))
    ctypes.CDLL(os.path.join(trt_libs_path, "libnvonnxparser.so.8"))
    ctypes.CDLL(os.path.join(trt_libs_path, "libnvparsers.so.8"))
    ctypes.CDLL(os.path.join(trt_libs_path, "libnvinfer_builder_resource.so.8.6.1"))
    import tensorrt as trt

# Initialize CUDA context before TensorRT
cuda_manager = CUDAContextManager()
cuda_manager.initialize()

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, "")


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class MyOutputAllocator(trt.IOutputAllocator):
    def __init__(self) -> None:
        super().__init__()
        # members for outside use
        self.shape = None
        self.n_bytes = 0
        self.address = 0

    def reallocate_output(self, tensor_name, old_address, size, alignment) -> int:
        return self.reallocate_common(tensor_name, old_address, size, alignment)

    def reallocate_output_async(
        self, tensor_name, old_address, size, alignment, stream
    ) -> int:
        return self.reallocate_common(tensor_name, old_address, size, alignment, stream)

    def notify_shape(self, tensor_name, shape):
        self.shape = shape
        return

    def reallocate_common(
        self, tensor_name, old_address, size, alignment, stream=-1
    ):  # not necessary API
        if size <= self.n_bytes:
            return old_address
        if old_address != 0:
            checkCudaErrors(cudart.cudaFree(old_address))
        if stream == -1:
            address = checkCudaErrors(cudart.cudaMalloc(size))
        else:
            address = checkCudaErrors(cudart.cudaMallocAsync(size, stream))
        self.n_bytes = size
        self.address = address
        return address


class TRTWrapper:
    def __init__(
        self,
        trt_file: str,
        plugin_file_list: list = [],
        high_priority: bool = False,
    ) -> None:
        # Load custom plugins
        for plugin_file in plugin_file_list:
            ctypes.cdll.LoadLibrary(plugin_file)

        # Load engine bytes from file
        self.model = trt_file
        with open(trt_file, "rb") as f, trt.Runtime(logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.buffer = OrderedDict()
        self.output_allocator_map = OrderedDict()
        self.context = self.engine.create_execution_context()
        
        # Cache for buffer reuse - tracks allocated shapes to avoid reallocation
        self._cached_input_shapes = {}
        self._buffers_initialized = False
        self._tensor_name_list = None
        self._n_input = None
        self._n_output = None
        
        # Create dedicated CUDA stream for this model
        if high_priority:
            _, lowest, highest = cudart.cudaDeviceGetStreamPriorityRange()
            _, self._cuda_stream = cudart.cudaStreamCreateWithPriority(
                cudart.cudaStreamNonBlocking, highest
            )
        else:
            _, self._cuda_stream = cudart.cudaStreamCreate()
        return

    def _init_tensor_info(self):
        """Initialize tensor info once (cached)."""
        if self._tensor_name_list is not None:
            return
        self._tensor_name_list = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self._n_input = sum(
            [
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                for name in self._tensor_name_list
            ]
        )
        self._n_output = self.engine.num_io_tensors - self._n_input

    def _shapes_match(self, input_data: dict) -> bool:
        """Check if input shapes match cached shapes."""
        if not self._buffers_initialized:
            return False
        for name, data in input_data.items():
            cached_shape = self._cached_input_shapes.get(name)
            if cached_shape is None or cached_shape != data.shape:
                return False
        return True

    def setup(self, input_data: dict = {}) -> None:
        # Initialize tensor info once
        self._init_tensor_info()
        
        # Fast path: if shapes match, just copy input data
        if self._shapes_match(input_data):
            for name, data in input_data.items():
                np.copyto(self.buffer[name][0], data)
            return
        
        # Slow path: need to reallocate buffers (only on shape change)
        # Free old buffers only if shapes changed
        for name, value in self.buffer.items():
            _, device_buffer, _ = value
            if (
                device_buffer is not None
                and device_buffer != 0
                and name not in self.output_allocator_map
            ):
                checkCudaErrors(cudart.cudaFree(device_buffer))
                self.buffer[name][1] = None
                self.buffer[name][2] = 0

        # For backwards compatibility
        self.tensor_name_list = self._tensor_name_list
        self.n_input = self._n_input
        self.n_output = self._n_output

        for name, data in input_data.items():
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_input_shape(name, data.shape)
            else:
                self.context.set_tensor_address(name, data.ctypes.data)
            # Cache the shape
            self._cached_input_shapes[name] = data.shape

        # Prepare work before inference
        for name in self._tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            if name not in self.output_allocator_map:
                if -1 in runtime_shape:
                    # for Data-Dependent-Shape (DDS) output, "else" branch for normal output
                    n_byte = 0  # self.context.get_max_output_size(name)
                    self.output_allocator_map[name] = MyOutputAllocator()
                    self.context.set_output_allocator(
                        name, self.output_allocator_map[name]
                    )
                    host_buffer = np.empty(0, dtype=trt.nptype(data_type))
                    device_buffer = None
                else:
                    n_byte = trt.volume(runtime_shape) * data_type.itemsize
                    host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
                    if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                        device_buffer = checkCudaErrors(cudart.cudaMalloc(n_byte))
                    else:
                        device_buffer = None
                self.buffer[name] = [host_buffer, device_buffer, n_byte]
            else:
                # for DDS output, don't need to reallocate
                pass

        for name, data in input_data.items():
            np.copyto(self.buffer[name][0], np.ascontiguousarray(data))

        for name in self._tensor_name_list:
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                if self.buffer[name][1] is not None:
                    self.context.set_tensor_address(name, self.buffer[name][1])
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(name, self.buffer[name][0].ctypes.data)

        self._buffers_initialized = True
        return

    def infer(self, stream=None) -> None:
        # Use dedicated stream if not specified
        if stream is None:
            stream = self._cuda_stream
        
        # with profile_block(f"{self.model}::infer"):
        # Copy inputs to device (async)
        for name in self._tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][1],
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )

        # Execute inference on stream
        self.context.execute_async_v3(stream)

        # Handle dynamic output shapes
        for name in self.output_allocator_map:
            myOutputAllocator = self.context.get_output_allocator(name)
            runtime_shape = myOutputAllocator.shape
            data_type = self.engine.get_tensor_dtype(name)
            host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
            device_buffer = myOutputAllocator.address
            n_bytes = trt.volume(runtime_shape) * data_type.itemsize
            self.buffer[name] = [host_buffer, device_buffer, n_bytes]

        # Copy outputs back to host (async)
        for name in self._tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][1],
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )

        # Synchronize with timeout to prevent infinite blocking
        self._sync_stream_with_timeout(stream, timeout_ms=5000)

        return
    
    def _sync_stream_with_timeout(self, stream, timeout_ms: int = 5000):
        """
        Synchronize stream with timeout to prevent infinite blocking.
        Uses cudaStreamQuery polling with sleep to avoid CPU spinning.
        """
        import time
        start_time = time.monotonic()
        timeout_sec = timeout_ms / 1000.0
        
        while True:
            result = cudart.cudaStreamQuery(stream)
            # cudaSuccess (0) means stream is complete
            if result[0] == cudart.cudaError_t.cudaSuccess:
                return
            # cudaErrorNotReady means still processing - this is expected
            elif result[0] == cudart.cudaError_t.cudaErrorNotReady:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout_sec:
                    # Timeout - force sync and log warning
                    logger.warning(f"Timeout waiting for stream {stream} to complete")
                    cudart.cudaStreamSynchronize(stream)
                    return
                # Sleep briefly to avoid CPU spinning (1ms)
                time.sleep(0.001)
            else:
                # Other error - force sync
                cudart.cudaStreamSynchronize(stream)
                return

    def infer_async(self, stream=0) -> None:
        # Do inference and print output
        for name in self._tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][1],
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream=stream,
                )

        self.context.execute_async_v3(stream)

        for name in self.output_allocator_map:
            myOutputAllocator = self.context.get_output_allocator(name)
            runtime_shape = myOutputAllocator.shape
            data_type = self.engine.get_tensor_dtype(name)
            host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
            device_buffer = myOutputAllocator.address
            n_bytes = trt.volume(runtime_shape) * data_type.itemsize
            self.buffer[name] = [host_buffer, device_buffer, n_bytes]

        for name in self._tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][1],
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream=stream,
                )

        return

    def __del__(self):
        # Clean up CUDA stream
        if hasattr(self, "_cuda_stream") and self._cuda_stream is not None and cudart is not None:
            try:
                cudart.cudaStreamDestroy(self._cuda_stream)
            except (TypeError, Exception):
                pass
        
        # Clean up device buffers
        if hasattr(self, "buffer") and self.buffer is not None:
            for _, device_buffer, _ in self.buffer.values():
                if (
                    device_buffer is not None
                    and device_buffer != 0
                    and cudart is not None
                ):
                    try:
                        checkCudaErrors(cudart.cudaFree(device_buffer))
                    except TypeError:
                        pass
        return
