import threading
from typing import Any, Dict, List
import time
import numpy as np
from loguru import logger
import asyncio

from .loader import load_source_frames
from .source2info import Source2Info
from ..utils.exceptions import UnsupportedSourceException

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


class SourceInfoManager:
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
        self.register_finish_event = asyncio.Event()

        self.source_info: Dict[str, Any] = {
            "x_s_info_lst": [],
            "f_s_lst": [],
            "M_c2o_lst": [],
            "eye_open_lst": [],
            "eye_ball_lst": [],
        }
        self.total_frames_count = 0
        self.cancel_regsiter_thread = threading.Event()
        self.pushed_entry_event = threading.Event()
        self.smo_k_s: int = 0
        self.source_gen_mutex = threading.Lock()
        self.register_thread: threading.Thread|None = None

    def reset_source_info(self):
        self.source_info = {
            "x_s_info_lst": [],
            "f_s_lst": [],
            "M_c2o_lst": [],
            "eye_open_lst": [],
            "eye_ball_lst": [],
        }
        

    def wait_until_index_ready(self, key: str, index: int):
        while len(self.source_info[key]) <= index:
            self.pushed_entry_event.wait()
            self.pushed_entry_event.clear()

    async def get_source_info_value_for_indices_list(
        self,
        key: str,
        indices: list[int]
    ):
        await asyncio.to_thread(self.wait_until_index_ready, key, max(indices))
        with self.source_gen_mutex:
            result = self.source_info[key][indices]
        return result
            
    async def get_source_info_value_for_index(
        self,
        key: str,
        index: int
    ):
        await asyncio.to_thread(self.wait_until_index_ready, key, index)
        with self.source_gen_mutex:
            result = self.source_info[key][index]
        return result

    def wait_for_loaded_source(self):
        while self.total_frames_count == 0:
            time.sleep(0.001)

    async def get_source_video_frame_count(self):
        await asyncio.to_thread(self.wait_for_loaded_source)
        return self.total_frames_count

    def get_source_info_component(self, key: str):
        result = None
        while result == None:
            time.sleep(0.001)
            with self.source_gen_mutex:
                result = self.source_info.get(key, None)
        return result
        

    def set_source_info(self, source_info: Dict[str, Any]):
        self.source_info = source_info

    def setup_source_info(
        self,
        rgb_frames: List[np.ndarray],
        is_image_flag: bool,
        **kwargs
    ):
        with self.source_gen_mutex:
            self.source_info["is_image_flag"] = is_image_flag
            self.source_info["img_rgb_lst"]   = rgb_frames

        keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
        last_lmk = None
        for rgb in rgb_frames:
            if self.cancel_regsiter_thread.is_set():
                return

            info = self.source2info(rgb, last_lmk, **kwargs)
            with self.source_gen_mutex:
                for k in keys:
                    self.source_info[f"{k}_lst"].append(info[k])

            if last_lmk is None:
                # first frame
                sc_f0 = self.source_info['x_s_info_lst'][0]['kp'].flatten()
                with self.source_gen_mutex:
                    self.source_info["sc"] = sc_f0

            last_lmk = info["lmk203"]
            self.pushed_entry_event.set()

        # final setup
        with self.source_gen_mutex:
            self.source_info["eye_open_lst"] = np.concatenate(self.source_info["eye_open_lst"], 0)  # [n, 2]
            self.source_info["eye_ball_lst"] = np.concatenate(self.source_info["eye_ball_lst"], 0)  # [n, 2]

            smooth_x_s_info_lst(
                x_s_info_list=self.source_info["x_s_info_lst"],
                smo_k=self.smo_k_s)


    def register(
        self,
        source_path,  # image | video
        max_dim=1920,
        n_frames=-1,
        **kwargs,
    ):
        """
        kwargs:
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        load_start_time = time.perf_counter()
        rgb_list, is_image_flag = load_source_frames(source_path, max_dim=max_dim, n_frames=n_frames)
        self.total_frames_count = len(rgb_list)
        load_end_time = time.perf_counter()
        logger.info(f"source video loading took: {load_end_time - load_start_time}s")
        self.setup_source_info(rgb_list, is_image_flag, **kwargs)

    async def __call__(self, *args, **kwargs):
        if self.register_thread is not None:
            self.cancel_regsiter_thread.set()
            self.register_thread.join()

        self.cancel_regsiter_thread.clear()
        self.reset_source_info()
        self.register_thread = threading.Thread(target=self.register, args=args, kwargs=kwargs)
        self.register_thread.start()
        #return await asyncio.to_thread(self.register, *args, **kwargs)

    