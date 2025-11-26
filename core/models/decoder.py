import numpy as np
import torch
from ..utils.load_model import load_model


class Decoder:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "SPADEDecoder",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device
        
    def __call__(self, feature):

        if self.model_type == "onnx":
            pred = self.model.run(None, {"feature": feature})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature": feature})
            self.model.infer()
            pred = self.model.buffer["output"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                pred = self.model(torch.from_numpy(feature).to(self.device)).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        pred = np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255    # [h, w, c]
        
        return pred

    def decode_batch(self, features_batch):
        """
        Decode a batch of features in a single forward pass.
        
        Args:
            features_batch: Batched features array with shape [batch_size, ...]
            
        Returns:
            List of decoded images
        """
        batch_size = features_batch.shape[0]
        
        if self.model_type == "onnx":
            preds = self.model.run(None, {"feature": features_batch})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature": features_batch})
            self.model.infer()
            preds = self.model.buffer["output"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True):
                preds = self.model(torch.from_numpy(features_batch).to(self.device)).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Process each prediction in the batch
        results = []
        for i in range(batch_size):
            pred = np.transpose(preds[i], [1, 2, 0]).clip(0, 1) * 255  # [h, w, c]
            results.append(pred)
        
        return results
