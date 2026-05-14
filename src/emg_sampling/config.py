"""Project-level constants for reproducible EMG experiments."""

PROJECT_CHANNEL_COUNT = 24
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
