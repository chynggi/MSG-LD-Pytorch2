"""Local copy of DISCoder components for 44.1 kHz vocoding.

The implementation is derived from https://github.com/ETH-DISCO/discoder
which is released under the MIT License. Only the modules required for
inference are mirrored here so that the MusicLDM pipeline can load the
pretrained 44.1 kHz vocoder from Hugging Face without adding the entire
repository as an external dependency.
"""

from .models import DisCoder

__all__ = ["DisCoder"]
