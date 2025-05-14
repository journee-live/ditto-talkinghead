import time

from loguru import logger


class FPSTracker:
    total_frames: int = 0
    total_frames_time: float = 0
    last_update_time: float = 0
    partial_frames: int = 0
    last_partial_time: float = 0
    total_partial_time: float = 0
    is_running: bool = False
    id: str = ""    

    def __init__(self, id: str):
        self.id = id

    def start(self):
        # This method should be called when the first frame is generated
        # We are assuming first frame takes exactly 40ms for a proper average. Ditto actually takes around 1s to generate the first frame
        self.last_update_time = time.monotonic() - 0.04
        self.total_frames = 0
        self.total_frames_time = 0

        self.partial_frames = 0
        self.last_partial_time = time.monotonic() - 0.04
        self.total_partial_time = 0

        self.is_running = True

    def stop(self):
        self.is_running = False

    def update(self, num_frames: int):
        self.total_frames += num_frames
        self.total_frames_time += time.monotonic() - self.last_update_time
        self.last_update_time = time.monotonic()

        self.partial_frames += num_frames
        self.total_partial_time += time.monotonic() - self.last_partial_time
        self.last_partial_time = time.monotonic()

    @property
    def average_fps(self) -> float:
        if self.total_frames <= 1:
            return 0

        return (self.total_frames) / (self.total_frames_time)

    @property
    def partial_average_fps(self) -> float:
        if self.partial_frames <= 1:
            return 0

        return (self.partial_frames) / (self.total_partial_time)

    def log(self):
        """Log current statistics."""
        logger.info(
            f"{self.id} : FPS={self.average_fps:.4f} PartialFPS: {self.partial_average_fps:.4f} total_frames:{self.total_frames} partial_frames:{self.partial_frames} "
        )
        self.partial_frames = 0
        self.total_partial_time = 0
