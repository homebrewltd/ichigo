import dataclasses
import math
import random
from typing import Optional


def rand(start: float, end: float) -> float:
    """Generate a random float between start and end."""
    return random.random() * (end - start) + start


def logrand(start: float, end: float) -> float:
    """Generate a random float between start and end in log space."""
    return 10 ** rand(math.log10(start), math.log10(end))


@dataclasses.dataclass
class VQConfig:
    """Configuration for Vector Quantization model.

    Attributes:
        # Initialization Parameters
        init_std (float): Standard deviation for weight initialization
        embeddings_std (float): Standard deviation for embeddings
        embeddings_lr_scale (float): Learning rate scaling for embeddings

        # Model Architecture
        query_mult (float): Multiplier for attention query scaling
        rope (bool): Whether to use Rotary Position Embedding
        mask_embs (bool): Whether to use masked embeddings
        output_mult (Optional[int]): Output multiplier

        # Downsampling Options
        downsample_conv (bool): Use convolution for downsampling
        downsample_mean (bool): Use mean pooling for downsampling

        # Vector Quantization Parameters
        codebook_dim (int): Dimension of each codebook vector
        codebook_decay (float): EMA decay rate for codebook updates

        # Training Parameters
        lr0 (float): Initial learning rate
        clip_gradient_norm (float): Gradient clipping threshold
        weight_decay (float): Weight decay for regularization
        warmup_steps (float): Number of warmup steps
        random (bool): Whether to use random parameter initialization
    """

    # Initialization Parameters
    init_std: float = 1.5
    embeddings_std: float = 4.5e-2
    embeddings_lr_scale: float = 1

    # Model Architecture
    query_mult: float = 2
    rope: bool = True
    mask_embs: bool = True
    output_mult: Optional[int] = None

    # Downsampling Options
    downsample_conv: bool = False
    downsample_mean: bool = True

    # Vector Quantization Parameters
    codebook_dim: int = 32
    codebook_decay: float = 0.9

    # Training Parameters
    lr0: float = 1e-3
    clip_gradient_norm: float = 2
    weight_decay: float = 1e-3
    warmup_steps: float = 500
    random: bool = False

    def __post_init__(self) -> None:
        """Initialize random parameters if specified."""
        if self.random:
            self._randomize_params()

    def _randomize_params(self) -> None:
        """Randomize model parameters within reasonable ranges."""
        # Initialization parameters
        self.init_std = logrand(1, 2)
        self.embeddings_std = logrand(3e-2, 6e-2)
        self.embeddings_lr_scale = 2 ** rand(0, 3)

        # Model architecture parameters
        self.query_mult = logrand(1, 8)
        self.codebook_dim = int(logrand(30, 50))

        # Training parameters
        self.codebook_decay = logrand(0.86, 0.95)
        self.lr0 = logrand(0.8e-3, 1e-3)
        self.clip_gradient_norm = 10 ** rand(-1, 1)
        self.warmup_steps = logrand(700, 1000)

    @staticmethod
    def upgrade(args: dict) -> dict:
        """Upgrade old configuration to new format.

        Args:
            args: Dictionary of configuration parameters

        Returns:
            Updated configuration dictionary with default values
        """
        args = {k: v for k, v in args.items()}

        # Default values for missing parameters
        defaults = {
            "output_mult": 1,
            "query_mult": 1,
            "rope": False,
            "mask_embs": False,
            "downsample_conv": False,
            "downsample_mean": False,
        }

        # Add missing defaults
        for k, v in defaults.items():
            if k not in args:
                args[k] = v

        # Remove deprecated parameters
        deprecated_params = ["encoder_depth_ratio", "vq_codes"]
        for k in deprecated_params:
            args.pop(k, None)

        return args
