import threading
from typing import Any, Dict, List
import time
import numpy as np
from loguru import logger
import asyncio
import os
from hashlib import md5

from dataclasses import dataclass, field

from spall_profiler import spall_profiler
from .source2info import Source2Info
from .loader import load_source_frames

DITTO_ROOT = os.path.dirname(os.path.abspath(__file__))

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

class Source_Info_Chunk:
    def __init__(self) -> None:
        self.data: List[Any] = []

class Source_Info_List_Chunks:
    def __init__(self) -> None:
        self.chunks: List[Source_Info_Chunk] = []
        self.total_count = 0
    
    def create_chunk(self):
        self.chunks.append(Source_Info_Chunk())

class Source_Info_Cache_Entry:
    def __init__(self) -> None:
        self.lists: Dict[str, Source_Info_List_Chunks] = {
            "x_s_info_lst": Source_Info_List_Chunks(),
            "f_s_lst": Source_Info_List_Chunks(),
            "M_c2o_lst": Source_Info_List_Chunks(),
            "eye_open_lst": Source_Info_List_Chunks(),
            "eye_ball_lst": Source_Info_List_Chunks(),
        }
        self.total_frames_count = 0
        self.sc = None
        self.condition = threading.Condition()

class SourceInfoCachingSystem:
    def __init__(self) -> None:
        self.source_info_cache: Dict[str, Source_Info_Cache_Entry] = {}
        self.new_cache_condition = threading.Condition()
        self.cache_dir: str = os.path.join(DITTO_ROOT, "source_info_cache")
        self.cache_version: str = field(init=True, default="0.1.0")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_avatar_id(self, source_path: str) -> str:
        avatar_id = "ditto_model." + source_path
        return md5(avatar_id.encode()).hexdigest()

    # Check if the avatar is in any level of the cache, and makes it available in faster caches
    def check_avatar_in_cache(self, source_path: str) -> bool:
        avatar_id = self.get_avatar_id(source_path)
        result = avatar_id in self.source_info_cache
        return result
    
    @spall_profiler.profile()
    def create_avatar_id_entry(self, avatar_id: str):
        with self.new_cache_condition:
            self.source_info_cache[avatar_id] = Source_Info_Cache_Entry()
            self.new_cache_condition.notify()
        
    @spall_profiler.profile()
    def wait_avatar_id_entry_creation(self, avatar_id: str):
        while True:
            with self.new_cache_condition:
                if avatar_id in self.source_info_cache:
                    break
                self.new_cache_condition.wait()

    @spall_profiler.profile()
    def register_frame_info(
        self, avatar_id: str, source_info: Dict[str, Any]
    ):
        if avatar_id not in self.source_info_cache:
            self.create_avatar_id_entry(avatar_id)
            sc_f0 = source_info['x_s_info']['kp'].flatten()
            self.source_info_cache[avatar_id].sc = sc_f0

        assert avatar_id in self.source_info_cache
        entry = self.source_info_cache[avatar_id]

        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        for k in keys:
            lst = entry.lists[f"{k}_lst"]
            if len(lst.chunks) == 0 or len(lst.chunks[-1].data) == MAX_FRAMES_PER_CHUNK:
                lst.create_chunk()
            chunk = lst.chunks[-1]
            chunk.data.append(source_info[k])
            lst.total_count += 1

        entry.total_frames_count += 1
        with entry.condition:
            entry.condition.notify()
    
    def serialize_list_chunks(self, avatar_id: str, key: str, lst: Source_Info_List_Chunks):
        # TODO: implement this
        return

    def finalize_avatar_registering(self, avatar_id: str, smo_k_s: int):
        logger.debug(f"source info final setup")
        assert avatar_id in self.source_info_cache

        entry = self.source_info_cache[avatar_id]
        for key in ["eye_open_lst", "eye_ball_lst"]:
            lst = entry.lists[key]
            for chunk in lst.chunks:
                chunk.data = np.concatenate(chunk.data, 0)  # [n, 2]
        
        for chunk in entry.lists["x_s_info_lst"].chunks:
            smooth_x_s_info_lst(
                x_s_info_list=chunk.data,
                smo_k=smo_k_s
            )

        # NOTE: save to disk
        for key in entry.lists:
            lst = entry.lists[key]
            self.serialize_list_chunks(avatar_id, key, lst)
    
    def get_source_info_dir(self, avatar_id: str):
        result = os.path.join(self.cache_dir, avatar_id)
        return result

    def try_load_from_disk(self, avatar_id: str) -> bool:
        result = False
        cache_path = self.get_source_info_dir(avatar_id)
        if os.path.exists(cache_path):
            result = True
            keys = ["x_s_info_lst", "f_s_lst", "M_c2o_lst", "eye_open_lst", "eye_ball_lst"]
        
        return result

    @spall_profiler.profile()
    def try_load_source_info(self, avatar_id: str) -> bool:
        result = False
        # NOTE: try load from disk
        if self.try_load_from_disk(avatar_id):
            result = True
                
        return result

        
    @spall_profiler.profile()
    def get_value_for_index(self, avatar_id: str, key: str, idx: int):
        self.wait_avatar_id_entry_creation(avatar_id)

        assert avatar_id in self.source_info_cache
        entry = self.source_info_cache[avatar_id]
        lst = entry.lists[key]

        value = None
        while value is None:
            idx_offset = 0
            for chunk in lst.chunks:
                if idx_offset + len(chunk.data) > idx:
                    value = chunk.data[idx - idx_offset]
                    break
                idx_offset += len(chunk.data)

            if value is None:
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
        self.avatar_id: str = ""
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
        self.avatar_id = source_info_caching.get_avatar_id(source_path)

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
        if not source_info_caching.try_load_source_info(self.avatar_id):
            last_lmk = None
            for idx, rgb in enumerate(self.rgb_list):
                info = self.source2info(rgb, last_lmk, **kwargs)
                source_info_caching.register_frame_info(self.avatar_id, info)
                last_lmk = info["lmk203"]
                logger.debug(f"generated source info entry: {idx}")
            source_info_caching.finalize_avatar_registering(self.avatar_id, self.smo_k_s)

    async def get_value_for_index(self, key: str, idx: int):
        result = await asyncio.to_thread(source_info_caching.get_value_for_index, self.avatar_id, key, idx)
        return result

    async def get_source_video_frame_count(self):
        return len(self.rgb_list)
