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


@dataclass
class Source_Info_Cache_Entry:
    data: Dict[str, Any]
    last_lmk: Any

class SourceInfoCachingSystem:
    def __init__(self):
        self.source_info_cache: Dict[str, Source_Info_Cache_Entry] = {}

    def get_avatar_id(self, source_path: str) -> str:
        avatar_id = "ditto_model." + source_path
        return md5(avatar_id.encode()).hexdigest()

    # Check if the avatar is in any level of the cache, and makes it available in faster caches
    def check_avatar_in_cache(self, source_path: str) -> bool:
        avatar_id = self.get_avatar_id(source_path)
        result = avatar_id in self.source_info_cache
        return result

    @spall_profiler.profile()
    def register_frame_info(
        self, avatar_id: str, source_info: Dict[str, Any]
    ):

        if avatar_id not in self.source_info_cache:
            self.source_info_cache[avatar_id] = Source_Info_Cache_Entry(
                data={
                    "x_s_info_lst": [],
                    "f_s_lst": [],
                    "M_c2o_lst": [],
                    "eye_open_lst": [],
                    "eye_ball_lst": [],
                }, 
                last_lmk=None
            )

            sc_f0 = source_info['x_s_info']['kp'].flatten()
            self.source_info_cache[avatar_id].data["sc"] = sc_f0

        assert avatar_id in self.source_info_cache

        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        for k in keys:
            self.source_info_cache[avatar_id].data[f"{k}_lst"].append(source_info[k])
    
    def finalize_avatar_registering(self, avatar_id: str, smo_k_s: int, last_lmk: Any):
        logger.debug(f"source info final setup")
        assert avatar_id in self.source_info_cache

        source_info = self.source_info_cache[avatar_id].data
        source_info["eye_open_lst"] = np.concatenate(source_info["eye_open_lst"], 0)  # [n, 2]
        source_info["eye_ball_lst"] = np.concatenate(source_info["eye_ball_lst"], 0)  # [n, 2]

        smooth_x_s_info_lst(
            x_s_info_list=source_info["x_s_info_lst"],
            smo_k=smo_k_s
        )
        self.source_info_cache[avatar_id].last_lmk = last_lmk
        

    def get_value_for_index(self, avatar_id: str, key: str, idx: int):
        assert avatar_id in self.source_info_cache
        source_info = self.source_info_cache[avatar_id].data
        value = source_info[key][idx]
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
    ):
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


    def register_avatar(
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
        self.rgb_list, self.is_image_flag = load_source_frames(source_path, max_dim=max_dim, n_frames=n_frames)

        loaded_from_cache = source_info_caching.check_avatar_in_cache(source_path)
        if not loaded_from_cache:
            self.generate_source_info(kwargs=kwargs)
        return loaded_from_cache
    
    def generate_source_info(self, **kwargs):
        last_lmk = None
        for idx, rgb in enumerate(self.rgb_list):
            info = self.source2info(rgb, last_lmk, **kwargs)
            source_info_caching.register_frame_info(self.avatar_id, info)
            last_lmk = info["lmk203"]
            logger.debug(f"generated source info entry: {idx}")
        source_info_caching.finalize_avatar_registering(self.avatar_id, self.smo_k_s, last_lmk)

    async def get_value_for_index(self, key: str, idx: int):
        result = source_info_caching.get_value_for_index(self.avatar_id, key, idx)
        return result

    async def get_source_video_frame_count(self):
        return len(self.rgb_list)
