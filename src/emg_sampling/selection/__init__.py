"""Channel-selection helpers for EMG experiments."""

from emg_sampling.selection.channel_ranking import rank_channels_by_macro_f1
from emg_sampling.selection.greedy_selection import greedy_channel_order
from emg_sampling.selection.random_selection import sample_random_channels

__all__ = [
    "greedy_channel_order",
    "rank_channels_by_macro_f1",
    "sample_random_channels",
]
