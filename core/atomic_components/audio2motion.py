import numpy as np

from spall_profiler import spall_profiler
from ..models.lmdm import LMDM

"""
lmdm_cfg = {
    "model_path": "",
    "device": "cuda",
    "motion_feat_dim": 265,
    "audio_feat_dim": 1024+35,
    "seq_frames": 80,
}
"""


def _cvt_LP_motion_info(inp, mode, ignore_keys=()):
    ks_shape_map = [
        ["scale", (1, 1), 1],
        ["pitch", (1, 66), 66],
        ["yaw", (1, 66), 66],
        ["roll", (1, 66), 66],
        ["t", (1, 3), 3],
        ["exp", (1, 63), 63],
        ["kp", (1, 63), 63],
    ]

    def _dic2arr(_dic):
        arr = []
        for k, _, ds in ks_shape_map:
            if k not in _dic or k in ignore_keys:
                continue
            v = _dic[k].reshape(ds)
            if k == "scale":
                v = v - 1
            arr.append(v)
        arr = np.concatenate(arr, -1)  # (133)
        return arr

    def _arr2dic(_arr):
        dic = {}
        s = 0
        for k, ds, ss in ks_shape_map:
            if k in ignore_keys:
                continue
            v = _arr[s : s + ss].reshape(ds)
            if k == "scale":
                v = v + 1
            dic[k] = v
            s += ss
            if s >= len(_arr):
                break
        return dic

    if mode == "dic2arr":
        assert isinstance(inp, dict)
        return _dic2arr(inp)  # (dim)
    elif mode == "arr2dic":
        assert inp.shape[0] >= 265, f"{inp.shape}"
        return _arr2dic(inp)  # {k: (1, dim)}
    else:
        raise ValueError()


class Audio2Motion:
    def __init__(
        self,
        lmdm_cfg,
    ):
        self.lmdm = LMDM(**lmdm_cfg)
        # Initialize storage for keypoint condition history
        self.kp_cond_history = []

    @spall_profiler.profile()
    def setup(
        self,
        x_s_info,
        overlap_v2=10,
        fix_kp_cond=0,
        fix_kp_cond_dim=None,
        sampling_timesteps=50,
        online_mode=False,
        v_min_max_for_clip=None,
        smo_k_d=3,
    ):
        self.filter_amount = 0.1
        self.smo_k_d = smo_k_d
        self.overlap_v2 = overlap_v2
        self.seq_frames = self.lmdm.seq_frames
        self.valid_clip_len = self.seq_frames - self.overlap_v2

        # for fuse
        self.online_mode = online_mode
        if self.online_mode:
            self.fuse_length = min(self.overlap_v2, self.valid_clip_len)
        else:
            self.fuse_length = self.overlap_v2
        self.fuse_alpha = (
            np.arange(self.fuse_length, dtype=np.float32).reshape(1, -1, 1)
            / self.fuse_length
        )

        self.fix_kp_cond = fix_kp_cond
        self.fix_kp_cond_dim = fix_kp_cond_dim
        self.sampling_timesteps = sampling_timesteps

        self.v_min_max_for_clip = v_min_max_for_clip
        if self.v_min_max_for_clip is not None:
            self.v_min = self.v_min_max_for_clip[0][None]  # [dim, 1]
            self.v_max = self.v_min_max_for_clip[1][None]

        kp_source = _cvt_LP_motion_info(x_s_info, mode="dic2arr", ignore_keys={"kp"})[
            None
        ]
        self.s_kp_cond = kp_source.copy().reshape(1, -1)
        self.kp_cond = self.s_kp_cond.copy()

        self.lmdm.setup(sampling_timesteps)

        self.clip_idx = 0

    def _fuse(self, res_kp_seq, pred_kp_seq):
        ## ========================
        ## offline fuse mode
        ## last clip:  -------
        ## fuse part:    *****
        ## curr clip:    -------
        ## output:       ^^
        #
        ## online fuse mode
        ## last clip:  -------
        ## fuse part:       **
        ## curr clip:    -------
        ## output:          ^^
        ## ========================

        fuse_r1_s = res_kp_seq.shape[1] - self.fuse_length
        fuse_r1_e = res_kp_seq.shape[1]
        fuse_r2_s = self.seq_frames - self.valid_clip_len - self.fuse_length
        fuse_r2_e = self.seq_frames - self.valid_clip_len

        r1 = res_kp_seq[:, fuse_r1_s:fuse_r1_e]  # [1, fuse_len, dim]
        r2 = pred_kp_seq[:, fuse_r2_s:fuse_r2_e]  # [1, fuse_len, dim]
        r_fuse = r1 * (1 - self.fuse_alpha) + r2 * self.fuse_alpha

        res_kp_seq[:, fuse_r1_s:fuse_r1_e] = r_fuse  # fuse last
        res_kp_seq = np.concatenate(
            [res_kp_seq, pred_kp_seq[:, fuse_r2_e:]], 1
        )  # len(res_kp_seq) + valid_clip_len

        return res_kp_seq

    def _update_kp_cond(self, res_kp_seq, idx):
        """
        Updates the keypoint condition values based on the current animation state.

        The keypoint condition (kp_cond) represents the current state of the avatar's
        facial expressions and head movements, which influences the next animation.
        This method controls how these condition values are updated between frames.

        Args:
            res_kp_seq: Keypoint sequence array [batch_size, sequence_length, feature_dim]
            idx: Index pointing to the frame to use for condition update

        Behavior depends on self.fix_kp_cond setting:
            - 0: Always update condition from the previous frame (continuous motion)
            - >0: Periodically reset to source condition every fix_kp_cond clips
                 This creates more varied motion by preventing drift over time
        """
        if self.fix_kp_cond == 0:  # No reset, continuous update mode
            # Take the previous frame as the new condition
            # This creates smooth transitions between frames
            self.kp_cond = res_kp_seq[:, idx - 1]
        elif self.fix_kp_cond > 0:
            if self.clip_idx % self.fix_kp_cond == 0:  # Reset on interval
                # Reset to source condition (original reference pose)
                self.kp_cond = self.s_kp_cond.copy()  # Reset all dimensions

                # If specific dimensions are configured for selective updating
                if self.fix_kp_cond_dim is not None:
                    # Extract dimension range to selectively update
                    ds, de = self.fix_kp_cond_dim
                    # Only update specified dimensions from the previous frame
                    # This allows partial resets where some features continue naturally
                    # while others return to reference values
                    self.kp_cond[:, ds:de] = res_kp_seq[:, idx - 1, ds:de]
            else:
                # On non-reset frames, update normally from previous frame
                self.kp_cond = res_kp_seq[:, idx - 1]

    def _smo(self, res_kp_seq, s, e):
        # Define which parameters correspond to facial expressions that should NOT be smoothed
        smo = int(self.smo_k_d * self.filter_amount * 30)
        if smo > 1:
            new_res_kp_seq = res_kp_seq.copy()
            n = res_kp_seq.shape[1]
            half_k = smo // 2

            # Apply standard smoothing first
            for i in range(s, e):
                ss = max(0, i - half_k)
                ee = min(n, i + half_k + 1)
                res_kp_seq[:, i, :202] = np.mean(new_res_kp_seq[:, ss:ee, :202], axis=1)

        return res_kp_seq

    # @profile("Audio2Motion Forward")
    @spall_profiler.profile("audio2motion")
    def __call__(self, aud_cond, res_kp_seq=None):
        """
        aud_cond: (1, seq_frames, dim)
        """

        pred_kp_seq = self.lmdm(self.kp_cond, aud_cond, self.sampling_timesteps)
        if res_kp_seq is None:
            res_kp_seq = pred_kp_seq  # [1, seq_frames, dim]
            # original_kp_seq = res_kp_seq.copy()
            res_kp_seq = self._smo(res_kp_seq, 0, res_kp_seq.shape[1])

        else:
            res_kp_seq = self._fuse(
                res_kp_seq, pred_kp_seq
            )  # len(res_kp_seq) + valid_clip_len
            # original_kp_seq = res_kp_seq.copy()
            res_kp_seq = self._smo(
                res_kp_seq,
                res_kp_seq.shape[1] - self.valid_clip_len - self.fuse_length,
                res_kp_seq.shape[1] - self.valid_clip_len + 1,
            )

        # if self.clip_idx <= 14:
        #     # Plot keypoint sequence comparison
        #     #plot_smoothing_in_pipeline(original_kp_seq, res_kp_seq, self.clip_idx)

        #     # Store keypoint condition values for later plotting
        #     if hasattr(self, "kp_cond") and self.kp_cond is not None:
        #         # Ensure we're storing a 1D array
        #         if len(self.kp_cond.shape) == 2 and self.kp_cond.shape[0] == 1:
        #             self.kp_cond_history.append(self.kp_cond[0].copy())
        #         else:
        #             self.kp_cond_history.append(self.kp_cond.copy())

        #     # Plot keypoint condition values over time when we reach the last clip
        #     if self.clip_idx == 14:
        #         from examples.plot_ditto_smoothing import plot_kp_cond_over_time

        #         plot_kp_cond_over_time(self.kp_cond_history, self.clip_idx)

        self.clip_idx += 1

        idx = res_kp_seq.shape[1] - self.overlap_v2
        self._update_kp_cond(res_kp_seq, idx)

        return res_kp_seq

    def cvt_fmt(self, res_kp_seq):
        # res_kp_seq: [1, n, dim]
        if self.v_min_max_for_clip is not None:
            tmp_res_kp_seq = np.clip(res_kp_seq[0], self.v_min, self.v_max)
        else:
            tmp_res_kp_seq = res_kp_seq[0]

        x_d_info_list = []
        for i in range(tmp_res_kp_seq.shape[0]):
            x_d_info = _cvt_LP_motion_info(tmp_res_kp_seq[i], "arr2dic")  # {k: (1, dim)}
            x_d_info_list.append(x_d_info)
        return x_d_info_list

    def reset_state(self):
        # Reset clip counter
        self.clip_idx = 0

        # Reset keypoint condition history
        self.kp_cond_history = []

        # Reset keypoint condition to source values
        if hasattr(self, "s_kp_cond"):
            self.kp_cond = self.s_kp_cond.copy()

        # Reset fuse parameters
        if self.online_mode:
            self.fuse_length = min(self.overlap_v2, self.valid_clip_len)
        else:
            self.fuse_length = self.overlap_v2

        self.fuse_alpha = (
            np.arange(self.fuse_length, dtype=np.float32).reshape(1, -1, 1)
            / self.fuse_length
        )

        # Reset min/max clipping values if they exist
        if self.v_min_max_for_clip is not None:
            self.v_min = self.v_min_max_for_clip[0][None]
            self.v_max = self.v_min_max_for_clip[1][None]
