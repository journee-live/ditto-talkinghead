import os

import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result
    

class PutBack:
    def __init__(
        self,
        mask_template_path=None,
        sharpen_amount: float | None = None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0]
        self.result_buffer = None
        # Allow override via environment variable (default 0.3 for subtle sharpening)
        if sharpen_amount is None:
            sharpen_amount = float(os.environ.get("DITTO_SPEECH_SHARPEN_AMOUNT", "0.0"))
        self.sharpen_amount = sharpen_amount
        
        # Pre-compute sharpening kernel (unsharp mask style)
        # This enhances edges to counteract blur from warpAffine and blending
        self._sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        
        # Use INTER_CUBIC for sharper results (slightly slower but much better quality)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)
        
        # TODO Improve performance since it tanks fps!!
        # Apply subtle sharpening to counteract blur from blending
        if self.sharpen_amount > 0:
            # Blend between original and sharpened result
            sharpened = cv2.filter2D(self.result_buffer, -1, self._sharpen_kernel)
            # Only sharpen the face region (where mask > 0.5) to avoid edge artifacts
            mask_3ch = np.repeat(mask_warped[:, :, np.newaxis], 3, axis=2)
            face_mask = (mask_3ch > 0.5).astype(np.float32)
            self.result_buffer = np.clip(
                self.result_buffer * (1 - face_mask * self.sharpen_amount) + 
                sharpened * (face_mask * self.sharpen_amount),
                0, 255
            ).astype(np.uint8)

        return self.result_buffer