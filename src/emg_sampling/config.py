"""Project-level constants for reproducible EMG experiments."""

PROJECT_CHANNEL_COUNT = 24
GRABMYO_SIGNAL_NAMES_32 = (
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "U1",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "U2",
    "U3",
    "W7",
    "W8",
    "W9",
    "W10",
    "W11",
    "W12",
    "U4",
)
GRABMYO_DEFAULT_CHANNEL_INDICES_32 = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
    25,
    26,
)
PROJECT_CHANNEL_NAMES_24 = tuple(
    GRABMYO_SIGNAL_NAMES_32[index] for index in GRABMYO_DEFAULT_CHANNEL_INDICES_32
)

TARGET_CLASSES = {
    "WF": "wrist flexion",
    "WE": "wrist extension",
    "HO": "hand open",
    "HC": "hand close",
}

TARGET_GESTURES = {
    11: "WF",
    12: "WE",
    15: "HO",
    16: "HC",
}

LABELS = {"WF": 0, "WE": 1, "HO": 2, "HC": 3}

WINDOW_MS_CANDIDATES = [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
DEFAULT_WINDOW_MS = 200.0
NOISE_WINDOW_MS_CANDIDATES = [150.0, 200.0, 250.0]
HOP_FRACTION = 0.25

NOTCH_HZ = 50.0
NOTCH_Q = 30.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
BANDPASS_ORDER = 4

N_TEST_PARTICIPANTS = 2
NOISE_BASE_SEED = 123

SPLIT_MODES = ("quick", "medium", "full")


def resolve_project_channel_indices(total_channels: int) -> tuple[int, ...]:
    """Returns the default project channel subset for a GRABMyo record."""
    if total_channels == PROJECT_CHANNEL_COUNT:
        return tuple(range(total_channels))
    if total_channels == 32:
        return GRABMYO_DEFAULT_CHANNEL_INDICES_32
    raise ValueError(
        "Unsupported GRABMyo channel layout. "
        f"Expected {PROJECT_CHANNEL_COUNT} or 32 channels, got {total_channels}."
    )


def resolve_project_channel_names(total_channels: int) -> tuple[str, ...]:
    """Returns signal names for the project channel subset."""
    if total_channels == PROJECT_CHANNEL_COUNT:
        return PROJECT_CHANNEL_NAMES_24
    if total_channels == 32:
        return PROJECT_CHANNEL_NAMES_24
    raise ValueError(
        "Unsupported GRABMyo channel layout. "
        f"Expected {PROJECT_CHANNEL_COUNT} or 32 channels, got {total_channels}."
    )
