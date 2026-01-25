from typing import Any, Dict
import numpy as np

class BaseDataset:
    """
    Base dataset class (follows a PyTorch-style)
    """

    def __len__(self) -> int:
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

