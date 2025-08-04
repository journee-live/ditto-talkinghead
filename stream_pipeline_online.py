import queue
import threading
import time
import traceback

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel

from .core.atomic_components.audio2motion import Audio2Motion
from .core.atomic_components.avatar_registrar import (
    AvatarRegistrar,
    smooth_x_s_info_lst,
)
from .core.atomic_components.cfg import parse_cfg, print_cfg
from .core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from .core.atomic_components.decode_f3d import DecodeF3D
from .core.atomic_components.motion_stitch import MotionStitch
from .core.atomic_components.putback import PutBack
from .core.atomic_components.warp_f3d import WarpF3D
from .core.atomic_components.wav2feat import Wav2Feat
from .core.utils.profiling_utils import FPSTracker
from .core.utils.threading_utils import AtomicCounter

"""
avatar_registrar_cfg:
    insightface_det_cfg,
    landmark106_cfg,
    landmark203_cfg,
    landmark478_cfg,
    appearance_extractor_cfg,
    motion_extractor_cfg,

condition_handler_cfg:
    use_emo=True,
    use_sc=True,
    use_eye_open=True,
    use_eye_ball=True,
    seq_frames=80,

wav2feat_cfg:
    w2f_cfg, 
    w2f_type
"""


class SDKDebugState(BaseModel):
    hubert_features_queue: int
    audio2motion_queue: int
    motion_stitch_queue: int
    putback_queue: int
    warp_f3d_queue: int
    decode_f3d_queue: int
    frame_queue: int
    pending_frames: int
    expected_frames: int
    starting_gen_frame_idx: int
    is_expecting_more_audio: bool
    hubert_finished: bool
    reset_audio2motion_needed: bool


class StreamSDK:
    def __init__(self, cfg_pkl, data_root, chunk_size, **kwargs):
        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)

        self.default_kwargs = default_kwargs
        self.BYTES_PER_FRAME = 640
        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()
        self.chunk_size = chunk_size
        self.overlap_size = self.chunk_size[0] * self.BYTES_PER_FRAME
        self.future_size = self.chunk_size[2] * self.BYTES_PER_FRAME
        self.present_size = self.chunk_size[1] * self.BYTES_PER_FRAME
        self.split_len = int(sum(self.chunk_size) * self.BYTES_PER_FRAME) + 80
        self.wav2feat = Wav2Feat(**wav2feat_cfg)
        self.starting_gen_frame_idx = 0
        self.motion_output_enabled = False
        self.idle_to_speech_transition = None
        self.fps_tracker = FPSTracker("streamSDK")
        self.start_processing_time = 0

        # Initialize the stop event for thread control
        self.stop_event = threading.Event()
        # Initialize reset flag
        self.reset_audio2motion_needed = threading.Event()
        self.reset_audio2motion_needed.set()
        self.is_expecting_more_audio = threading.Event()

        self.hubert_finished = threading.Event()
        self.hubert_finished.set()
        self.worker_exception = None
        self.waiting_for_first_audio = True
        self.expected_frames = AtomicCounter(0)
        self.pending_frames = AtomicCounter(0)
        # Create threads for pipeline stages
        self.thread_list = []

    def _start_threads(self):
        """Initialize and start all worker threads for the pipeline"""
        self.close()
        self.stop_event.clear()
        # Reset all parameters and queues
        self.reset()

        # Clear the thread list
        self.thread_list = []

        # Create all worker threads
        self.thread_list = [
            threading.Thread(
                target=self.hubert_worker, name="hubert_worker", daemon=False
            ),
            threading.Thread(
                target=self.audio2motion_worker,
                name="audio2motion_worker",
                daemon=False,
            ),
            threading.Thread(
                target=self.motion_stitch_worker,
                name="motion_stitch_worker",
                daemon=False,
            ),
            threading.Thread(
                target=self.warp_f3d_worker, name="warp_f3d_worker", daemon=False
            ),
            threading.Thread(
                target=self.decode_f3d_worker, name="decode_f3d_worker", daemon=False
            ),
            threading.Thread(
                target=self.putback_worker, name="putback_worker", daemon=False
            ),
        ]

        # Start all threads
        for thread in self.thread_list:
            thread.start()

    def _merge_kwargs(self, default_kwargs, run_kwargs):
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs

    def setup_Nd(
        self,
        N_d,
        fade_in=-1,
        fade_out=-1,
        ctrl_info=None,
        fade_in_frame_offset=0,
        fade_out_frame_offset=0,
    ):
        # for eye open at video end
        self.motion_stitch.set_Nd(N_d)
        self.motion_stitch.fade_out_keys = ctrl_info["fade_keys"]

        # Helper function for ease in/out transitions
        def ease_in_out(t):
            """Cubic ease in/out function: smoother transitions at both ends"""
            if t < 0.5:
                # Ease in (cubic): 4 * t^3
                return 4 * t * t * t
            else:
                # Ease out (cubic): 1 - 4 * (1-t)^3
                return 1 - 4 * (1 - t) * (1 - t) * (1 - t)

        # for fade in/out alpha
        if ctrl_info is None:
            ctrl_info = dict()

        # Apply fade-in transition
        if fade_in > 0:
            for i in range(fade_in_frame_offset, fade_in_frame_offset + fade_in):
                t = (i - fade_in_frame_offset) / (fade_in)
                alpha = ease_in_out(t)  # Apply ease in/out function
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item

        # Apply fade-out transition
        if fade_out > 0:
            ss = N_d - fade_out - fade_out_frame_offset
            ee = N_d - 1 - fade_out_frame_offset
            for i in range(ss, N_d):
                t = 1.0 - max((i - ss) / (ee - ss), 0)  # Reverse time for fade-out
                alpha = ease_in_out(t)  # Apply ease in/out function
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item

        self.ctrl_info = ctrl_info

    def setup(self, source_path, source_info=None, **kwargs):
        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        print("=" * 20, "setup kwargs", "=" * 20)
        print_cfg(**kwargs)
        print("=" * 50)

        # -- avatar_registrar: template cfg --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)

        # -- avatar_registrar: crop cfg --
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)

        # -- avatar_registrar: smo for video --
        self.smo_k_s = kwargs.get("smo_k_s", 13)

        # -- condition_handler: ECS --
        self.emo = kwargs.get("emo", 4)  # int | [int] | [[int]] | numpy
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)  # for video
        self.ch_info = kwargs.get("ch_info", None)  # dict of np.ndarray

        # -- audio2motion: setup --
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)  # [ds,de]
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get("v_min_max_for_clip", None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)

        # -- motion_stitch: setup --
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)  # None: true4image, false4video
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")  # "" | "d0" | "s"
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)

        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        """
        ctrl_info: list or dict
            {
                fid: ctrl_kwargs
            }

            ctrl_kwargs (see motion_stitch.py):
                fade_alpha
                fade_out_keys

                delta_pitch
                delta_yaw
                delta_roll
        """

        # only hubert support online mode
        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Register Avatar ========
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        if source_info is None:
            source_info = self.avatar_registrar(
                source_path,
                max_dim=self.max_size,
                n_frames=n_frames,
                **crop_kwargs,
            )

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(
                source_info["x_s_info_lst"], smo_k=self.smo_k_s
            )

        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])

        # ======== Setup Condition Handler ========
        self.condition_handler.setup(
            source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info
        )

        # ======== Setup Audio2Motion (LMDM) ========
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0,
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )

        # ======== Setup Motion Stitch ========
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info["x_s_info_lst"][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

        self.initial_audio_feat = self.wav2feat.wav2feat(
            np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000
        )
        # ======== Audio Feat Buffer ========
        self.reset_audio_features()
        # ======== Setup Worker Threads ========
        QUEUE_MAX_SIZE = 0
        # self.QUEUE_TIMEOUT = None

        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_out_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.frame_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.hubert_features_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self._start_threads()

    def _get_ctrl_info(self, fid):
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception:
            traceback.print_exc()
            return {}

    def putback_worker(self):
        try:
            self._putback_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _putback_worker(self):
        last_frame_time = time.monotonic()
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:
                self.putback_queue.task_done()
                continue

            frame_idx, render_img, gen_frame_idx = item
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o = self.source_info["M_c2o_lst"][frame_idx]
            res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            self.pending_frames.decrement(1)

            # logger.debug(
            #     f"Generated video frame dt: {time.monotonic() - last_frame_time:.4f}"
            # )
            last_frame_time = time.monotonic()
            frame_bgr = cv2.cvtColor(res_frame_rgb, cv2.COLOR_RGB2BGR)

            # Encode frame to JPEG
            success, frame_data = cv2.imencode(".jpg", frame_bgr)

            if not self.fps_tracker.is_running:
                self.fps_tracker.start()

            self.fps_tracker.update(1)

            if self.fps_tracker.total_frames == 1:
                logger.info(
                    f"Time until first frame: {time.monotonic() - self.start_processing_time}"
                )

            if gen_frame_idx % 25 == 0:
                self.fps_tracker.log()

            self.frame_queue.put([frame_data.tobytes(), frame_idx, gen_frame_idx])
            self.putback_queue.task_done()

    def decode_f3d_worker(self):
        try:
            self._decode_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:
                self.decode_f3d_queue.task_done()
                continue

            frame_idx, f_3d, gen_frame_idx = item
            render_img = self.decode_f3d(f_3d)
            self.putback_queue.put([frame_idx, render_img, gen_frame_idx])
            self.decode_f3d_queue.task_done()

    def warp_f3d_worker(self):
        try:
            self._warp_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=0.05)
                # Clear the flag when we have work to do
            except queue.Empty:
                continue

            if item is None:
                self.warp_f3d_queue.task_done()
                continue

            frame_idx, x_s, x_d, gen_frame_idx = item
            f_s = self.source_info["f_s_lst"][frame_idx]
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            self.decode_f3d_queue.put([frame_idx, f_3d, gen_frame_idx])
            self.warp_f3d_queue.task_done()

    def motion_stitch_worker(self):
        # try:
        self._motion_stitch_worker()

    # except Exception as e:
    #     self.worker_exception = e
    #     self.stop_event.set()

    def _motion_stitch_worker(self):
        num_frames_stitched = 0
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=0.05)
                # Clear the flag when we have work to do
            except queue.Empty:
                continue

            if item is None:
                self.motion_stitch_queue.task_done()
                continue

            frame_idx, x_d_info, ctrl_kwargs, gen_frame_idx = item
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]
            if gen_frame_idx > self.expected_frames.get() + self.starting_gen_frame_idx:
                self.motion_stitch_queue.task_done()
                continue

            if (
                gen_frame_idx
                == self.expected_frames.get() + self.starting_gen_frame_idx - 1
            ):
                logger.info("Last frame on motion stitch worker!!!")
            x_s, x_d, x_d_info = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            num_frames_stitched += 1
            if self.motion_output_enabled:
                self.motion_stitch_out_queue.put([x_d_info, frame_idx, gen_frame_idx])
            self.warp_f3d_queue.put([frame_idx, x_s, x_d, gen_frame_idx])
            self.motion_stitch_queue.task_done()

    def hubert_worker(self):
        try:
            self._hubert_worker()
        except Exception as e:
            print("Error in hubert_worker:", e)
            traceback.print_exc()
            self.worker_exception = e
            self.stop_event.set()

    def _hubert_worker(self):
        chunk_idx = 0
        while not self.stop_event.is_set():
            if self.reset_audio2motion_needed.is_set():
                continue
            if self.warp_f3d_queue.qsize() > 30:
                time.sleep(0.05)
                continue

            # if self.fps_tracker.average_fps > 30:
            #     # Make a pause in frame generation so we don't overload memory with frames for long answers
            #     time.sleep(1.0)

            try:
                audio_chunk = self.hubert_features_queue.get(timeout=0.05)  # audio feat
                # Clear the finished flag as we have work to do
                self.hubert_finished.clear()
            except queue.Empty:
                self.hubert_finished.set()
                continue

            if audio_chunk is None:
                continue

            # Process audio through HuBERT
            item = self.wav2feat(audio_chunk, chunksize=self.chunk_size)

            # Put the processed features in the queue
            self.audio2motion_queue.put((item, chunk_idx))
            self.hubert_features_queue.task_done()
            chunk_idx += 1

    def audio2motion_worker(self):
        try:
            self._audio2motion_worker()            
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def interrupt(self):
        # We don't interrupt something that has already been interrupted before
        if self.waiting_for_first_audio:
            return

        self.waiting_for_first_audio = True
        logger.info("Interrupting all workers")
        self.end_processing_audio()

        # Restart the threads
        self._start_threads()

    def log_queues(self):
        logger.info(self.get_debug_state().model_dump_json(indent=4))

    def get_debug_state(self):
        return SDKDebugState(
            hubert_features_queue=self.hubert_features_queue.qsize(),
            audio2motion_queue=self.audio2motion_queue.qsize(),
            motion_stitch_queue=self.motion_stitch_queue.qsize(),
            putback_queue=self.putback_queue.qsize(),
            warp_f3d_queue=self.warp_f3d_queue.qsize(),
            decode_f3d_queue=self.decode_f3d_queue.qsize(),
            frame_queue=self.frame_queue.qsize(),
            pending_frames=self.pending_frames.get(),
            expected_frames=self.expected_frames.get(),
            starting_gen_frame_idx=self.starting_gen_frame_idx,
            is_expecting_more_audio=self.is_expecting_more_audio.is_set(),
            hubert_finished=self.hubert_finished.is_set(),
            reset_audio2motion_needed=self.reset_audio2motion_needed.is_set(),
        )

    def _audio2motion_worker(self):
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item = None
        is_end = False
        processing_frames = 0
        all_audio_processed = True
        audio_feat = self.initial_audio_feat
        local_idx = 0
        started_processing = False
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)
        while not self.stop_event.is_set():
            if all_audio_processed:
                self.reset_audio2motion_needed.wait()

            try:
                # reset audio2motion to generate new answers
                if self.reset_audio2motion_needed.is_set():
                    logger.info(
                        f"Resetting audio2motion starting gen_frame_idx: {self.starting_gen_frame_idx}"
                    )
                    self.reset_audio2motion_needed.clear()
                    global_idx = 0  # frame idx, for template
                    local_idx = 0  # for cur audio_feat
                    gen_frame_idx = self.starting_gen_frame_idx
                    started_processing = False
                    res_kp_seq = None
                    is_end = False
                    processing_frames = 0
                    all_audio_processed = False
                    item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)
                    res_kp_seq_valid_start = None if self.online_mode else 0
                    audio_feat = self.initial_audio_feat
                    cond_idx_start = 0 - len(audio_feat)
                    assert len(audio_feat) == self.overlap_v2, f"{len(audio_feat)}"

                is_end = (
                    started_processing
                    and not self.is_expecting_more_audio.is_set()
                    and self.hubert_finished.is_set()
                    and self.audio2motion_queue.qsize() == 0
                )
                if not is_end:
                    item, chunk_idx = self.audio2motion_queue.get(
                        timeout=0.05
                    )  # audio feat
                    if not started_processing:
                        logger.info("Starting processing audio2motion")
                        started_processing = True

            except queue.Empty:
                # logger.info(f"Audio2Motion queue is empty, is_end={is_end}")
                # IF queue is empty and we expect more audio we wait until it comes
                item = None
                if not is_end:
                    # logger.warning("Audio2Motion queue is empty before ending")
                    continue

            # We don't have anything else to do
            if is_end:
                all_audio_processed = True

            # This prevents a crash when placing breakpoints on other threads
            if item is not None:
                item_buffer = np.concatenate([item_buffer, item], 0)
                processing_frames += len(item)

            if not is_end and item_buffer.shape[0] < valid_clip_len:
                # wait at least valid_clip_len new item
                # logger.info("Waiting for more audio features")
                continue
            else:
                audio_feat = np.concatenate([audio_feat, item_buffer], 0)
                item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            logger.debug(
                f"Processing new frames batch processing frames={processing_frames} is_end={is_end}"
            )

            while True:
                if self.stop_event.is_set():
                    return

                aud_feat_slice = audio_feat[
                    local_idx : local_idx + seq_frames
                ]  # typically 80 frames slice
                real_valid_len = valid_clip_len
                if len(aud_feat_slice) == 0:
                    break

                elif len(aud_feat_slice) < seq_frames:
                    # logger.info(
                    #     f"Audio feature length {len(aud_feat_slice)} is less than seq_frames {seq_frames}"
                    # )
                    if not is_end:
                        # wait next chunk
                        break
                    else:
                        # final clip: pad to seq_frames
                        real_valid_len = len(aud_feat_slice)
                        pad = np.stack(
                            [aud_feat_slice[-1]] * (seq_frames - len(aud_feat_slice)), 0
                        )
                        aud_feat_slice = np.concatenate([aud_feat_slice, pad], 0)

                aud_cond = self.condition_handler(
                    aud_feat_slice, global_idx + cond_idx_start
                )[None]
                # self.audio2motion.clip_idx = (
                #     gen_frame_idx / self.audio2motion.valid_clip_len
                # )
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                if res_kp_seq_valid_start is None:
                    # online mode, first chunk
                    res_kp_seq_valid_start = (
                        res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    )
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0

                    local_idx += real_valid_len
                    global_idx += real_valid_len
                    continue
                else:
                    valid_res_kp_seq = res_kp_seq[
                        :,
                        res_kp_seq_valid_start : res_kp_seq_valid_start
                        + real_valid_len,
                    ]
                    x_d_info_list = self.audio2motion.cvt_fmt(valid_res_kp_seq)

                    for x_d_info in x_d_info_list:
                        # early exit if stop event is set to avoid having to wait
                        if self.stop_event.is_set():
                            return

                        frame_idx = _mirror_index(
                            gen_frame_idx, self.source_info_frames
                        )
                        ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                        if (
                            gen_frame_idx
                            < self.expected_frames.get() + self.starting_gen_frame_idx
                        ):
                            self.motion_stitch_queue.put(
                                [frame_idx, x_d_info, ctrl_kwargs, gen_frame_idx],
                                timeout=0.1,
                            )
                        else:
                            logger.info("No more frames expected from audio2motion!")
                            self.reset_audio2motion_needed.set()
                            break

                        gen_frame_idx += 1

                    res_kp_seq_valid_start += real_valid_len

                    local_idx += real_valid_len
                    global_idx += real_valid_len

                L = res_kp_seq.shape[1]
                if seq_frames * 2 < L:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(audio_feat):
                    break

            L = len(audio_feat)
            if seq_frames * 2 < L:
                cut_L = L - seq_frames * 2
                audio_feat = audio_feat[cut_L:]
                local_idx -= cut_L

        self.motion_stitch_queue.put(None)

    def close(self):
        # flush frames
        self.stop_event.set()
        self.reset()

        # Wait for all threads to finish
        for thread in self.thread_list:
            thread.join()

        # Check if any worker encountered an exception
        if self.worker_exception is not None:
            raise self.worker_exception

    def reset(self):
        logger.info("reset")
        self.fps_tracker.stop()
        self.motion_stitch.reset_state()
        self.audio2motion.reset_state()

        # Clear all queues
        self.audio2motion_queue.queue.clear()
        self.motion_stitch_queue.queue.clear()
        self.putback_queue.queue.clear()
        self.warp_f3d_queue.queue.clear()
        self.decode_f3d_queue.queue.clear()
        self.frame_queue.queue.clear()
        self.hubert_features_queue.queue.clear()
        self.motion_stitch_out_queue.queue.clear()

        self.expected_frames.set(0)
        self.pending_frames.set(0)
        self.reset_audio_features()

    def reset_audio_features(self):
        self.reset_audio2motion_needed.set()

    def start_processing_audio(
        self, start_frame_idx: int = 0, filter_amount: float = 0.0, mouth_opening_scale: float = 1.0
    ):
        logger.info("start_processing_audio")
        self.starting_gen_frame_idx = start_frame_idx
        self.audio2motion.filter_amount = filter_amount
        self.motion_stitch.mouth_opening_scale = mouth_opening_scale
        self.start_processing_time = time.monotonic()
        self.is_expecting_more_audio.set()

    def set_motion_output_enabled(self, motion_output_enabled: bool = False):
        self.motion_output_enabled = motion_output_enabled

    def end_processing_audio(self):
        logger.info("end_processing_audio")
        self.is_expecting_more_audio.clear()

    def process_audio_chunk(self, audio_chunk):
        # Clear all finished flags when new work is submitted
        self.hubert_finished.clear()

        # Process audio
        self.hubert_features_queue.put(audio_chunk)

        self.expected_frames.increment(self.chunk_size[1])
        self.pending_frames.increment(self.chunk_size[1])

    def process_audio(
        self, audio: np.ndarray, pad_audio: bool = False, max_frames: int = -1
    ) -> np.ndarray:
        self.waiting_for_first_audio = False

        # Process each chunk
        processed_audio_idx = 0
        for idx in range(0, len(audio), self.present_size):
            # Extract the chunk
            audio_chunk = audio[idx : idx + self.split_len]

            # Pad last chunk if needed
            if len(audio_chunk) < self.split_len:
                if not pad_audio:
                    break
                audio_chunk = np.pad(
                    audio_chunk,
                    (0, self.split_len - len(audio_chunk)),
                    mode="constant",
                )
                processed_audio_idx = idx + len(audio_chunk)
            else:
                processed_audio_idx = idx + self.present_size
            # Process chunk and yield frames
            self.process_audio_chunk(audio_chunk)
            if (
                max_frames > 0
                and processed_audio_idx * self.chunk_size[1] >= max_frames
            ):
                break

        audio = audio[processed_audio_idx:]

        if processed_audio_idx > 0:
            logger.debug(
                f"Processed {processed_audio_idx} audio samples pending frames {self.pending_frames.get()} expected frames {self.expected_frames.get()}"
            )

        return audio

    def has_pending_frames(self):
        pending_frames = self.pending_frames.get()
        frame_queue_size = self.frame_queue.qsize()
        return (
            pending_frames > 0
            or frame_queue_size > 0
            or self.is_expecting_more_audio.is_set()
        )
