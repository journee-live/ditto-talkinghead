import threading
from typing import Any, Dict, List, Mapping
import time
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

class SourceInfoCachingSystem:
    def __init__(self) -> None:
        self.source_info_cache: Dict[str, Source_Info_Cache_Entry] = {}
        self.new_cache_condition = threading.Condition()
        self.cache_dir: str = os.path.join(DITTO_ROOT, "source_info_cache")
        self.cache_version: str = field(init=True, default="0.1.0")
        self.chunk_load_request: deque[str] = deque()
        self.workers = ThreadPoolExecutor(max_workers=4)
        self.memory_cache_max_size: int = settings.MAX_CACHE_INFO_SIZE_GB
        self.current_cache_size: int = 0

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_source_id(self, source_path: str) -> str:
        source_id = "ditto_model." + source_path
        return md5(source_id.encode()).hexdigest()

    # Check if the avatar is in any level of the cache, and makes it available in faster caches
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

    @spall_profiler.profile()
    def estimate_size_bytes(self, obj, seen: set[int] | None = None) -> int:
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
                size += self.estimate_size_bytes(k, seen)
                size += self.estimate_size_bytes(v, seen)
            return size

        # Iterables (list/tuple/set) but not strings/bytes
        if isinstance(obj, (list, tuple, set, frozenset)):
            size = sys.getsizeof(obj)
            for item in obj:
                size += self.estimate_size_bytes(item, seen)
            return size

        # Fallback
        return sys.getsizeof(obj)

    def evict_cache_until_size(self, ignore_id: str, target_size: int):
        while self.current_cache_size > target_size:
            # TODO: better eviction strategy? right now we just pick at random
            if len(self.source_info_cache.keys()) == 0:
               break
            random_key = random.choice(list(self.source_info_cache))
            if random_key != ignore_id:
                random_entry = self.source_info_cache[random_key]
                self.current_cache_size -= random_entry.size
                self.source_info_cache.pop(random_key)

    @spall_profiler.profile()
    def register_frame_info(
        self, source_id: str, source_info: Dict[str, Any]
    ):
        data_size = self.estimate_size_bytes(source_info)
        logger.info(f"registering new source info frame, size: {data_size}")
        if self.current_cache_size + data_size > self.memory_cache_max_size:
            self.evict_cache_until_size(
                ignore_id=source_id,
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
        # TODO(mouad): queue it for background worker thread
        joblib.dump(chunk, chunk_path, compress=0, protocol=pickle.HIGHEST_PROTOCOL)

    @spall_profiler.profile()
    def queue_chunk_disk_load_request(self, chunk_path: str):
        self.chunk_load_request.append(chunk_path)

    @spall_profiler.profile()
    def queue_get_next_loaded_chunk(self) -> Dict[str, list]|None:
        result = None
        if len(self.chunk_load_request) > 0:
            chunk_path = self.chunk_load_request.popleft()
            result = joblib.load(chunk_path)
        return result

    @spall_profiler.profile()
    def serialize_entry(self, source_id: str, entry: Source_Info_Cache_Entry):
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
            self.queue_chunk_disk_write_request(chunk=chunk, chunk_path=chunk_path)

        return

    @spall_profiler.profile()
    def load_and_deserialize(self, source_id: str, info_cache_dir: str):
        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        chunk_idx = 0
        while True:
            chunk_path = os.path.join(info_cache_dir, f"chunk-{chunk_idx}.pkl")
            chunk_idx += 1
            if not os.path.exists(chunk_path):
                break
            self.queue_chunk_disk_load_request(chunk_path=chunk_path)
        
        while True:
            chunk = self.queue_get_next_loaded_chunk()
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

        # TODO: avoid loading this all the time
        self.rgb_list, self.is_image_flag = await asyncio.to_thread(load_source_frames, source_path, max_dim=max_dim, n_frames=n_frames)

        loaded_from_cache = source_info_caching.check_avatar_in_cache(source_path)
        if not loaded_from_cache:
            self.src_info_gen_thread = threading.Thread(target=self.generate_source_info, kwargs=kwargs)
            self.src_info_gen_thread.start()
            #self.generate_source_info(kwargs=kwargs)

        return loaded_from_cache

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
