import numpy as np
import torch

from ..utils.load_model import load_model
from ..utils.profile_util import profile


def make_beta(n_timestep, cosine_s=8e-3):
    timesteps = (
        torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
    )
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return betas.numpy()


class LMDM:
    @profile("LMDM::init")
    def __init__(self, model_path, device="cuda", **kwargs):
        kwargs["module_name"] = "LMDM"

        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.motion_feat_dim = kwargs.get("motion_feat_dim", 265)
        self.audio_feat_dim = kwargs.get("audio_feat_dim", 1024+35)
        self.seq_frames = kwargs.get("seq_frames", 80)

        if self.model_type == "pytorch":
            pass
        else:
            self._init_np()

    @profile("LMDM::setup")
    def setup(self, sampling_timesteps):
        if self.model_type == "pytorch":
            self.model.setup(sampling_timesteps)
        else:
            self._setup_np(sampling_timesteps)

    @profile("LMDM::_init_np")
    def _init_np(self):
        self.sampling_timesteps = None
        self.n_timestep = 1000

        betas = torch.Tensor(make_beta(n_timestep=self.n_timestep))
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cpu().numpy()

    @profile("LMDM::_setup_np")
    def _setup_np(self, sampling_timesteps=50):
        if self.sampling_timesteps == sampling_timesteps:
            return
        
        self.sampling_timesteps = sampling_timesteps

        total_timesteps = self.n_timestep
        eta = 1
        shape = (1, self.seq_frames, self.motion_feat_dim)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(
            zip(times[:-1], times[1:], strict=False)
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        self.time_cond_list = []
        self.alpha_next_sqrt_list = []
        self.sigma_list = []
        self.c_list = []
        self.noise_list = []

        for time, time_next in self.time_pairs:
            time_cond = np.full((1,), time, dtype=np.int64)
            self.time_cond_list.append(time_cond)
            if time_next < 0:
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = np.sqrt(1 - alpha_next - sigma ** 2)
            noise = np.random.randn(*shape).astype(np.float32)
            
            self.alpha_next_sqrt_list.append(np.sqrt(alpha_next))
            self.sigma_list.append(sigma)
            self.c_list.append(c)
            self.noise_list.append(noise)

    @profile("lmdm::_one_step")
    def _one_step(self, x, cond_frame, cond, time_cond):
        if self.model_type == "onnx":
            pred = self.model.run(None, {"x": x, "cond_frame": cond_frame, "cond": cond, "time_cond": time_cond})
            pred_noise, x_start = pred[0], pred[1]
        elif self.model_type == "tensorrt":
            # Use pinned memory for faster transfers
            with torch.cuda.stream(torch.cuda.Stream()):
                inputs = {
                    "x": torch.from_numpy(x).cuda(non_blocking=True),
                    "cond_frame": torch.from_numpy(cond_frame).cuda(non_blocking=True),
                    "cond": torch.from_numpy(cond).cuda(non_blocking=True),
                    "time_cond": torch.from_numpy(time_cond).cuda(non_blocking=True),
                }
                # Convert back to CPU numpy arrays for TensorRT
                inputs = {name: tensor.cpu().numpy() for name, tensor in inputs.items()}
                self.model.setup(inputs)
                # Optional: Enable FP16 if supported by your model
                # self.model.set_precision(trt.float16)
                self.model.infer()
                pred_noise = self.model.buffer["pred_noise"][0]
                x_start = self.model.buffer["x_start"][0]
                # Ensure computations are complete
                torch.cuda.current_stream().synchronize()
        elif self.model_type == "pytorch":
            with torch.no_grad():
                pred_noise, x_start = self.model(x, cond_frame, cond, time_cond)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred_noise, x_start

    def _call_np(self, kp_cond, aud_cond, sampling_timesteps):
        self._setup_np(sampling_timesteps)

        cond_frame = kp_cond
        cond = aud_cond

        x = np.random.randn(1, self.seq_frames, self.motion_feat_dim).astype(np.float32)

        x_start = None
        i = 0
        for _, time_next in self.time_pairs:
            time_cond = self.time_cond_list[i]
            pred_noise, x_start = self._one_step(x, cond_frame, cond, time_cond)
            if time_next < 0:
                x = x_start
                continue

            alpha_next_sqrt = self.alpha_next_sqrt_list[i]
            c = self.c_list[i]
            sigma = self.sigma_list[i]
            noise = self.noise_list[i]
            x = x_start * alpha_next_sqrt + c * pred_noise + sigma * noise

            i += 1

        return x

    @profile("LMDM::call")
    def __call__(self, kp_cond, aud_cond, sampling_timesteps):
        if self.model_type == "pytorch":
            pred_kp_seq = self.model.ddim_sample(
                torch.from_numpy(kp_cond).to(self.device), 
                torch.from_numpy(aud_cond).to(self.device), 
                sampling_timesteps,
            ).cpu().numpy()
        else:
            pred_kp_seq = self._call_np(kp_cond, aud_cond, sampling_timesteps)
        return pred_kp_seq


