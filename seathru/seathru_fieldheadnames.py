from enum import Enum


class SeathruHeadNames(Enum):
    """Additional field outputs for SeaThru-NeRF."""

    MEDIUM_RGB = "medium_RGB"
    MEDIUM_BS = "medium_bs"
    MEDIUM_ATTN = "medium_attn"
