import shutil
import threading
from typing import Any, Dict, List, Mapping
import numpy as np
from loguru import logger
import asyncio
import os
from hashlib import md5
import pickle
import joblib
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import random
from pathlib import Path

from dataclasses import dataclass, field

from spall_profiler import spall_profiler
from .source2info import Source2Info
from .loader import load_source_frames

from ....ditto_paths import DITTO_ROOT
from server.settings import settings

def _mean_filter(arr, k):
    n = arr.shape[0]
    half_k = k // 2
    res = [
        arr[max(0, i - half_k):min(n, i + half_k + 1)].mean(0) 
        for i in range(n)
    ]
    # for i in range(n):
    #     s = max(0, i - half_k)
    #     e = min(n, i + half_k + 1)
    #     res.append(arr[s:e].mean(0))
    res = np.stack(res, 0)
    return res

def smooth_x_s_info_lst(x_s_info_list, ignore_keys=(), smo_k=13):
    keys = x_s_info_list[0].keys()
    N = len(x_s_info_list)
    smo_dict = {}
    for k in keys:
        _lst = [x_s_info_list[i][k] for i in range(N)]
        if k not in ignore_keys:
            _lst = np.stack(_lst, 0)
            _smo_lst = _mean_filter(_lst, smo_k)
        else:
            _smo_lst = _lst
        smo_dict[k] = _smo_lst

    smo_res = [
        {k: smo_dict[k][i] for k in keys}
        for i in range(N)
    ]
    return smo_res


MAX_FRAMES_PER_CHUNK = 25

class Source_Info_List:
    def __init__(self) -> None:
        self.data : List[Any] = []
        self.total_count = 0
    
class Source_Info_Cache_Entry:
    def __init__(self) -> None:
        self.lists: Dict[str, Source_Info_List] = {
            "x_s_info_lst": Source_Info_List(),
            "f_s_lst": Source_Info_List(),
            "M_c2o_lst": Source_Info_List(),
            "eye_open_lst": Source_Info_List(),
            "eye_ball_lst": Source_Info_List(),
        }
        self.total_frames_count = 0
        self.sc = None
        self.condition = threading.Condition()
        self.size: int = 0

@spall_profiler.profile()
def estimate_size_bytes(obj, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)

    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.nbytes

    # Mappings (dict-like)
    if isinstance(obj, Mapping):
        size = sys.getsizeof(obj)
        for k, v in obj.items():
            size += estimate_size_bytes(k, seen)
            size += estimate_size_bytes(v, seen)
        return size

    # Iterables (list/tuple/set) but not strings/bytes
    if isinstance(obj, (list, tuple, set, frozenset)):
        size = sys.getsizeof(obj)
        for item in obj:
            size += estimate_size_bytes(item, seen)
        return size

    # Fallback
    return sys.getsizeof(obj)

@dataclass
class SourceVideoEntry:
    rgb_list: List[Any]
    is_image_flag: bool
    size: int
    max_dim: int
    n_frames: int

class SourceVideoCachingSystem:
    def __init__(self) -> None:
        self.video_frames: Dict[str, SourceVideoEntry] = {}
        self.current_size: int = 0
        self.max_size: int = settings.MAX_CACHE_SOURCE_FRAMES_SIZE_GB * 1024 * 1024 * 1024
    
    def evict_until_size(self, target_size: int):
        while self.current_size > target_size:
            random_key = random.choice(list(self.video_frames))
            logger.debug(f"evicting video cache entry for id: {random_key}")
            random_entry = self.video_frames[random_key]
            self.current_size -= random_entry.size
            self.video_frames.pop(random_key)

    async def get_source_rgb_list(self, id: str, max_dim: int, n_frames: int) -> tuple[List[Any], bool]:
        result = None
        if id in self.video_frames:
            entry = self.video_frames[id]
            if entry.max_dim == max_dim and entry.n_frames == n_frames:
                result = (entry.rgb_list, entry.is_image_flag)

        if result is None:
            if id in self.video_frames:
                entry = self.video_frames.pop(id)
                self.current_size -= entry.size

            rgb_list, is_image_flag = await asyncio.to_thread(load_source_frames, id, max_dim=max_dim, n_frames=n_frames)
            rgb_list_size = estimate_size_bytes(rgb_list)
            if self.current_size + rgb_list_size > self.max_size:
                self.evict_until_size(target_size=self.max_size - rgb_list_size)

            self.video_frames[id] = SourceVideoEntry(
                rgb_list=rgb_list,
                is_image_flag=is_image_flag,
                max_dim=max_dim,
                n_frames=n_frames,
                size=rgb_list_size
            )
            result = (rgb_list, is_image_flag)
            
        return result


class SourceInfoCachingSystem:
    def __init__(self) -> None:
        self.active_source_ids: set[str] = set()
        self.source_info_cache: Dict[str, Source_Info_Cache_Entry] = {}
        self.new_cache_condition = threading.Condition()
        self.cache_dir: str = os.path.join(DITTO_ROOT, "source_info_cache")
        self.cache_version: str = field(init=True, default="0.1.0")
        self.chunk_load_request: deque[str] = deque()
        self.workers = ThreadPoolExecutor(max_workers=4)
        self.memory_cache_max_size: int = settings.MAX_CACHE_INFO_SIZE_GB * 1024 * 1024 * 1024
        self.disk_cache_max_size: int = settings.MAX_CACHE_INFO_DISK_SIZE_GB * 1024 * 1024 * 1024
        self.current_cache_size: int = 0
        self.available_disk_chunk: Dict[int, Any] = {}
        self.new_disk_chunk_condition = threading.Condition()

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_source_id(self, source_path: str) -> str:
        source_id = "ditto_model." + source_path
        return md5(source_id.encode()).hexdigest()

    def set_active_id(self, source_id: str):
        self.active_source_ids.add(source_id)

    def remove_active_id(self, source_id: str):
        if source_id in self.active_source_ids:
            self.active_source_ids.remove(source_id)

    # Check if the avatar is in any memory cache
    def check_avatar_in_cache(self, source_path: str) -> bool:
        source_id = self.get_source_id(source_path)
        result = source_id in self.source_info_cache
        return result

    @spall_profiler.profile()
    def create_source_id_entry(self, source_id: str):
        with self.new_cache_condition:
            self.source_info_cache[source_id] = Source_Info_Cache_Entry()
            self.new_cache_condition.notify()

    @spall_profiler.profile()
    def wait_source_id_entry_creation(self, source_id: str):
        while True:
            with self.new_cache_condition:
                if source_id in self.source_info_cache:
                    break
                self.new_cache_condition.wait()

    def clean_cache_from_disk(self, source_dir: str) -> None:
        """
        Safely delete a cache directory or file from the file system.

        Args:
            source_path (str): Path to the cache file or directory to delete.
            source_dir (str): Path to the parent directory containing the cache file or directory.

        Raises:
            ValueError: If source path is invalid or empty
            PermissionError: If insufficient permissions to delete
            OSError: If deletion fails for other reasons

        """
        cache_dir = Path(source_dir).resolve()

        try:
            # Check if path exists
            if not cache_dir.exists():
                logger.info(f"Cache path does not exist, skipping: {cache_dir}")
                return

            # Check if we have permission to delete
            if not os.access(cache_dir.parent, os.W_OK):
                raise PermissionError(
                    f"No write permission for parent directory: {cache_dir.parent}"
                )

            try:
                # Attempt to remove directory and all contents
                shutil.rmtree(cache_dir)
                logger.info(f"Successfully deleted cache directory: {cache_dir}")
            except OSError:
                # If shutil.rmtree fails, try manual deletion for better error reporting
                for root, dirs, files in os.walk(cache_dir, topdown=False):
                    for file in files:
                        os.unlink(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(cache_dir)
                logger.info(
                    f"Successfully deleted cache directory (manual): {cache_dir}"
                )
            else:
                logger.warning(f"Unknown path type, skipping: {cache_dir}")

        except PermissionError as e:
            error_msg = f"Permission denied when deleting cache: {cache_dir} - {e!s}"
            logger.error(error_msg)
            raise PermissionError(error_msg) from e

        except FileNotFoundError:
            # File was deleted between existence check and deletion attempt
            logger.info(f"Cache path was already deleted: {cache_dir}")

        except OSError as e:
            error_msg = f"Failed to delete cache path: {cache_dir} - {e!s}"
            logger.error(error_msg)
            raise OSError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error deleting cache: {cache_dir} - {e!s}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @spall_profiler.profile()
    def evict_cache_until_size(self, target_size: int):
        while self.current_cache_size > target_size:
            # TODO: better eviction strategy? right now we just pick at random
            random_key = random.choice(list(self.source_info_cache))
            # check if we should stop
            should_stop = True
            for key in self.source_info_cache:
                if key not in self.active_source_ids:
                    should_stop = False
                    break
            if should_stop:
                break
            if random_key not in self.active_source_ids:
                logger.debug(f"evicting cache entry for id: {random_key}")
                random_entry = self.source_info_cache[random_key]
                self.current_cache_size -= random_entry.size
                self.source_info_cache.pop(random_key)

    @spall_profiler.profile()
    def evict_cache_disk_entry(self, source_id: str):
        entry_path = self.get_source_info_dir(source_id)
        try:
            self.clean_cache_from_disk(entry_path)
        except:
            logger.debug(f"Couldn't delete cache disk entry: {entry_path}")

    @spall_profiler.profile()
    def evict_cache_from_disk_until_size(self, target_size: int):
        disk_dir_size = self.get_directory_size(self.cache_dir)
        source_ids = [d for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))]
        while disk_dir_size > target_size:
            random_key = random.choice(source_ids)
            should_stop = True
            for key in source_ids:
                if key not in self.active_source_ids:
                    should_stop = False
                    break
            if should_stop:
                break
            if random_key not in self.active_source_ids:
                logger.debug(f"evicting disk cache entry for id: {random_key}")
                entry_path = self.get_source_info_dir(random_key)
                entry_path_size = self.get_directory_size(entry_path)
                disk_dir_size -= entry_path_size
                self.workers.submit(self.evict_cache_disk_entry, random_key)
                source_ids.remove(random_key)

    @spall_profiler.profile()
    def register_frame_info(
        self, source_id: str, source_info: Dict[str, Any]
    ):
        data_size = estimate_size_bytes(source_info)
        logger.info(f"registering new source info frame, size: {data_size}")
        if self.current_cache_size + data_size > self.memory_cache_max_size:
            self.evict_cache_until_size(
                target_size=self.memory_cache_max_size - data_size
            )

        if source_id not in self.source_info_cache:
            self.create_source_id_entry(source_id)
            sc_f0 = source_info['x_s_info']['kp'].flatten()
            self.source_info_cache[source_id].sc = sc_f0

        assert source_id in self.source_info_cache
        entry = self.source_info_cache[source_id]

        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        for k in keys:
            lst = entry.lists[f"{k}_lst"]
            lst.data.append(source_info[k])
            lst.total_count += 1

        self.current_cache_size += data_size
        entry.size += data_size
        entry.total_frames_count += 1
        with entry.condition:
            entry.condition.notify()

    @spall_profiler.profile()
    def queue_chunk_disk_write_request(self, chunk: Dict[str, list], chunk_path: str):
        joblib.dump(chunk, chunk_path, compress=0, protocol=pickle.HIGHEST_PROTOCOL)

    @spall_profiler.profile()
    def load_cache_chunk(self, index: int, chunk_path: str):
        result = joblib.load(chunk_path)
        self.available_disk_chunk[index] = result
        with self.new_disk_chunk_condition:
            self.new_disk_chunk_condition.notify()
    
    @spall_profiler.profile()
    def queue_chunk_disk_load_request(self, index: int, chunk_path: str):
        self.chunk_load_request.append(chunk_path)
        self.workers.submit(self.load_cache_chunk, index=index, chunk_path=chunk_path)

    @spall_profiler.profile()
    def queue_get_next_loaded_chunk(self, index: int) -> Dict[str, list]|None:
        result = None
        while True:
            if index in self.available_disk_chunk:
                result = self.available_disk_chunk.pop(index)
                break

            with self.new_disk_chunk_condition:
                if index not in self.available_disk_chunk:
                    self.new_disk_chunk_condition.wait()

        return result


    @spall_profiler.profile()
    def get_directory_size(self, path: str):
        return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    

    @spall_profiler.profile()
    def serialize_entry(self, source_id: str, entry: Source_Info_Cache_Entry):
        disk_dir_size = self.get_directory_size(self.cache_dir)
        if disk_dir_size + entry.size > self.disk_cache_max_size:
            self.evict_cache_from_disk_until_size(
                target_size=self.disk_cache_max_size - entry.size
            )

        info_cache_dir = self.get_source_info_dir(source_id)
        if not os.path.exists(info_cache_dir):
            os.makedirs(info_cache_dir)

        chunks_count = (entry.total_frames_count + MAX_FRAMES_PER_CHUNK - 1) // MAX_FRAMES_PER_CHUNK
        for i in range(chunks_count):
            offset = i * MAX_FRAMES_PER_CHUNK
            chunk: Dict[str, list] = {
                "x_s_info_lst": [],
                "f_s_lst": [],
                "M_c2o_lst": [],
                "eye_open_lst": [],
                "eye_ball_lst": [],
            }

            # build the chunk
            for key in entry.lists:
                chunk[key] = entry.lists[key].data[offset:offset+MAX_FRAMES_PER_CHUNK]

            chunk_path = os.path.join(info_cache_dir, f"chunk-{i}.pkl")
            self.workers.submit(self.queue_chunk_disk_write_request, chunk=chunk, chunk_path=chunk_path)

        return

    @spall_profiler.profile()
    def load_and_deserialize(self, source_id: str, info_cache_dir: str):
        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        chunk_idx = 0
        while True:
            chunk_path = os.path.join(info_cache_dir, f"chunk-{chunk_idx}.pkl")
            if not os.path.exists(chunk_path):
                break
            self.queue_chunk_disk_load_request(chunk_path=chunk_path, index=chunk_idx)
            chunk_idx += 1

        chunk_idx = 0
        while True:
            chunk = self.queue_get_next_loaded_chunk(chunk_idx)
            chunk_idx += 1
            if chunk is None:
                break

            frames_count_in_chunk = len(chunk["x_s_info_lst"])
            for idx in range(frames_count_in_chunk):
                source_info: Dict[str, Any] = {
                }
                for k in keys:
                    source_info[k] = chunk[f"{k}_lst"][idx]
                self.register_frame_info(source_id=source_id, source_info=source_info)

    @spall_profiler.profile()
    def try_load_from_disk(self, source_id: str, smo_k_s: int) -> bool:
        result = False
        info_cache_dir = self.get_source_info_dir(source_id)
        if os.path.exists(info_cache_dir):
            result = True
            self.load_and_deserialize(source_id, info_cache_dir)

            # TODO(mouad): check that we loaded all the needed frames, if we're still missing frames then
            #              generate the rest, maybe include last_lmk in each chunk so that we 
            #              can continue the generation from the last successful chunk

            self.post_process_source_data(source_id, smo_k_s)
        return result

    @spall_profiler.profile()
    def post_process_source_data(self, source_id: str, smo_k_s: int):
        assert source_id in self.source_info_cache
        entry = self.source_info_cache[source_id]

        for key in ["eye_open_lst", "eye_ball_lst"]:
            lst = entry.lists[key]
            lst.data = np.concatenate(lst.data, 0)  # [n, 2]
        
        smooth_x_s_info_lst(
            x_s_info_list=entry.lists["x_s_info_lst"].data,
            smo_k=smo_k_s
        )
        
    def finalize_avatar_registering(self, source_id: str, smo_k_s: int):
        logger.debug(f"source info final setup")
        assert source_id in self.source_info_cache
        entry = self.source_info_cache[source_id]

        # NOTE: save to disk
        self.serialize_entry(source_id, entry)
        self.post_process_source_data(source_id, smo_k_s)

    
    def get_source_info_dir(self, source_id: str):
        result = os.path.join(self.cache_dir, source_id)
        return result


    @spall_profiler.profile()
    def try_load_source_info(self, source_id: str, smo_k_s: int) -> bool:
        result = False
        # NOTE: try load from disk
        if self.try_load_from_disk(source_id, smo_k_s):
            result = True
                
        return result

    @spall_profiler.profile()
    def get_value_for_index(self, source_id: str, key: str, idx: int):
        self.wait_source_id_entry_creation(source_id)

        assert source_id in self.source_info_cache
        entry = self.source_info_cache[source_id]
        lst = entry.lists[key]

        value = None
        while value is None:
            if idx < len(lst.data):
                value = lst.data[idx]
                break

            with entry.condition:
                entry.condition.wait()

        return value


source_info_caching = SourceInfoCachingSystem()
source_video_caching = SourceVideoCachingSystem()

class SourceInfoGenerator:
    def __init__(
        self,
        insightface_det_cfg: Any,
        landmark106_cfg: Any,
        landmark203_cfg: Any,
        landmark478_cfg: Any,
        appearance_extractor_cfg: Any,
        motion_extractor_cfg: Any,
    ) -> None:
        self.source2info = Source2Info(
            insightface_det_cfg,
            landmark106_cfg,
            landmark203_cfg,
            landmark478_cfg,
            appearance_extractor_cfg,
            motion_extractor_cfg,
        )
        self.smo_k_s: int = 0
        self.source_id: str = ""
        self.rgb_list: list = []
        self.is_image_flag = False


    @spall_profiler.profile()
    async def register_avatar(
        self,
        smo_k_s: int,
        source_path: str,
        max_dim: int = 1920,
        n_frames: int = -1,
        **kwargs
    ) -> bool:
        self.smo_k_s = smo_k_s
        self.source_id = source_info_caching.get_source_id(source_path)
        source_info_caching.set_active_id(self.source_id)

        self.rgb_list, self.is_image_flag = await source_video_caching.get_source_rgb_list(id=source_path, max_dim=max_dim, n_frames=n_frames)

        loaded_from_cache = source_info_caching.check_avatar_in_cache(source_path)
        if not loaded_from_cache:
            self.src_info_gen_thread = threading.Thread(target=self.generate_source_info, kwargs=kwargs)
            self.src_info_gen_thread.start()
            #self.generate_source_info(kwargs=kwargs)

        return loaded_from_cache
    
    def unregister_avatar(self):
        source_info_caching.remove_active_id(self.source_id)

    @spall_profiler.profile()
    def generate_source_info(self, **kwargs):
        if not source_info_caching.try_load_source_info(self.source_id, self.smo_k_s):
            last_lmk = None
            for idx, rgb in enumerate(self.rgb_list):
                info = self.source2info(rgb, last_lmk, **kwargs)
                source_info_caching.register_frame_info(self.source_id, info)
                last_lmk = info["lmk203"]
                logger.debug(f"generated source info entry: {idx}")
            source_info_caching.finalize_avatar_registering(self.source_id, self.smo_k_s)
        return

    async def get_value_for_index(self, key: str, idx: int):
        result = await asyncio.to_thread(source_info_caching.get_value_for_index, self.source_id, key, idx)
        return result

    async def get_source_video_frame_count(self):
        return len(self.rgb_list)
