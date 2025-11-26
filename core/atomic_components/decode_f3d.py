import numpy as np

from ..models.decoder import Decoder


"""
# __init__
decoder_cfg = {
    "model_path": "",
    "device": "cuda",
}
"""

class DecodeF3D:
    def __init__(
        self,
        decoder_cfg,
    ):
        self.decoder = Decoder(**decoder_cfg)

    def __call__(self, f_s):
        out = self.decoder(f_s)
        return out

    def decode_batch(self, f_s_list: list) -> list:
        """
        Decode a batch of features in a single forward pass for better GPU utilization.
        
        Args:
            f_s_list: List of feature arrays to decode
            
        Returns:
            List of decoded images in the same order as input
        """
        if len(f_s_list) == 0:
            return []
        
        if len(f_s_list) == 1:
            return [self.decoder(f_s_list[0])]
        
        # Stack features into a batch
        batch = np.concatenate(f_s_list, axis=0)
        
        # Decode the batch
        results = self.decoder.decode_batch(batch)
        
        return results
    