"""
Synthetic test data generators for breath waveforms and sessions.

Provides functions to generate controlled, reproducible test data for unit testing.
"""

import numpy as np


def generate_sinusoidal_breath(
    duration: float = 4.0,
    amplitude: float = 30.0,
    sample_rate: float = 25.0,
    baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a perfect sinusoidal breath waveform.

    Args:
        duration: Breath duration in seconds
        amplitude: Peak flow amplitude in L/min
        sample_rate: Sample rate in Hz
        baseline: Baseline offset in L/min

    Returns:
        Tuple of (timestamps, flow_values)
    """
    n_samples = int(duration * sample_rate)
    timestamps = np.linspace(0, duration, n_samples)

    # Sinusoidal breath: positive (inspiration), negative (expiration)
    flow_values = amplitude * np.sin(2 * np.pi * timestamps / duration)
    flow_values += baseline

    return timestamps, flow_values


def generate_noisy_breath(
    duration: float = 4.0,
    amplitude: float = 30.0,
    sample_rate: float = 25.0,
    snr_db: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a breath waveform with controlled noise level.

    Args:
        duration: Breath duration in seconds
        amplitude: Peak flow amplitude in L/min
        sample_rate: Sample rate in Hz
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Tuple of (timestamps, flow_values)
    """
    timestamps, clean_signal = generate_sinusoidal_breath(
        duration, amplitude, sample_rate
    )

    # Calculate noise power from SNR
    signal_power = np.mean(clean_signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, len(clean_signal))
    noisy_signal = clean_signal + noise

    return timestamps, noisy_signal


def generate_flattened_breath(
    duration: float = 4.0,
    amplitude: float = 30.0,
    sample_rate: float = 25.0,
    flatness_index: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a flow-limited breath with specified flatness.

    Creates a waveform with flattened inspiratory phase to simulate
    flow limitation.

    Args:
        duration: Breath duration in seconds
        amplitude: Peak flow amplitude in L/min
        sample_rate: Sample rate in Hz
        flatness_index: Target flatness (0=sinusoidal, 1=perfectly flat)

    Returns:
        Tuple of (timestamps, flow_values)
    """
    n_samples = int(duration * sample_rate)
    timestamps = np.linspace(0, duration, n_samples)

    # Split into inspiration and expiration
    half = n_samples // 2

    # Inspiration: flatten the top portion
    insp_t = np.linspace(0, np.pi, half)
    insp_flow = amplitude * np.sin(insp_t)

    # Apply flattening: values above threshold become plateau
    plateau_threshold = amplitude * (1 - flatness_index)
    insp_flow = np.where(insp_flow > plateau_threshold, amplitude, insp_flow)

    # Expiration: normal sinusoidal
    exp_t = np.linspace(0, np.pi, n_samples - half)
    exp_flow = -amplitude * 0.7 * np.sin(exp_t)

    flow_values = np.concatenate([insp_flow, exp_flow])

    return timestamps, flow_values


def generate_multi_peak_breath(
    duration: float = 4.0,
    amplitude: float = 30.0,
    sample_rate: float = 25.0,
    peak_count: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a breath with multiple peaks during inspiration.

    Simulates flow limitation patterns like double-peak or vibration.

    Args:
        duration: Breath duration in seconds
        amplitude: Peak flow amplitude in L/min
        sample_rate: Sample rate in Hz
        peak_count: Number of peaks during inspiration

    Returns:
        Tuple of (timestamps, flow_values)
    """
    n_samples = int(duration * sample_rate)
    timestamps = np.linspace(0, duration, n_samples)

    # Split into inspiration and expiration
    half = n_samples // 2

    # Inspiration: create multiple peaks
    insp_t = np.linspace(0, np.pi, half)
    base_insp = amplitude * np.sin(insp_t)

    # Add oscillation to create multiple peaks
    oscillation = 0.3 * amplitude * np.sin(peak_count * np.pi * insp_t / np.pi)
    insp_flow = base_insp + oscillation

    # Expiration: normal
    exp_t = np.linspace(0, np.pi, n_samples - half)
    exp_flow = -amplitude * 0.7 * np.sin(exp_t)

    flow_values = np.concatenate([insp_flow, exp_flow])

    return timestamps, flow_values


def create_session(
    num_breaths: int = 30,
    avg_duration: float = 4.0,
    duration_variability: float = 0.5,
    avg_amplitude: float = 30.0,
    amplitude_variability: float = 5.0,
    sample_rate: float = 25.0,
    breath_type: str = "sinusoidal",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a complete session with multiple breaths.

    Args:
        num_breaths: Number of breaths to generate
        avg_duration: Average breath duration in seconds
        duration_variability: Std dev of duration variation
        avg_amplitude: Average breath amplitude in L/min
        amplitude_variability: Std dev of amplitude variation
        sample_rate: Sample rate in Hz
        breath_type: Type of breath ("sinusoidal", "flattened", "multi_peak")

    Returns:
        Tuple of (timestamps, flow_values) for entire session
    """
    all_timestamps = []
    all_flow_values = []
    current_time = 0.0

    for _i in range(num_breaths):
        # Vary duration and amplitude
        duration = np.random.normal(avg_duration, duration_variability)
        duration = max(1.0, duration)  # Minimum 1 second

        amplitude = np.random.normal(avg_amplitude, amplitude_variability)
        amplitude = max(10.0, amplitude)  # Minimum 10 L/min

        # Generate breath based on type
        if breath_type == "sinusoidal":
            t, flow = generate_sinusoidal_breath(duration, amplitude, sample_rate)
        elif breath_type == "flattened":
            t, flow = generate_flattened_breath(duration, amplitude, sample_rate)
        elif breath_type == "multi_peak":
            t, flow = generate_multi_peak_breath(duration, amplitude, sample_rate)
        else:
            raise ValueError(f"Unknown breath type: {breath_type}")

        # Offset timestamps to current time
        t = t + current_time
        current_time = t[-1]

        all_timestamps.extend(t)
        all_flow_values.extend(flow)

    return np.array(all_timestamps), np.array(all_flow_values)


def add_noise(
    waveform: np.ndarray,
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Add Gaussian noise to a waveform.

    Args:
        waveform: Clean waveform signal
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy waveform
    """
    signal_power = np.mean(waveform**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(0, noise_std, len(waveform))
    return waveform + noise


def add_artifacts(
    waveform: np.ndarray,
    timestamps: np.ndarray,
    artifact_percent: float = 5.0,
    artifact_type: str = "spike",
) -> np.ndarray:
    """
    Add artifacts to a waveform to simulate sensor issues.

    Args:
        waveform: Clean waveform signal
        timestamps: Corresponding timestamps
        artifact_percent: Percentage of samples to corrupt
        artifact_type: Type of artifact ("spike", "dropout", "nan")

    Returns:
        Waveform with artifacts
    """
    corrupted = waveform.copy()
    n_artifacts = int(len(waveform) * artifact_percent / 100)
    artifact_indices = np.random.choice(len(waveform), n_artifacts, replace=False)

    if artifact_type == "spike":
        # Large spikes (sensor glitches)
        corrupted[artifact_indices] = np.random.uniform(-150, 150, n_artifacts)
    elif artifact_type == "dropout":
        # Zero values (sensor disconnection)
        corrupted[artifact_indices] = 0.0
    elif artifact_type == "nan":
        # NaN values (data corruption)
        corrupted[artifact_indices] = np.nan
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")

    return corrupted


def create_multi_segment_session(
    segment_count: int = 3,
    breaths_per_segment: int = 20,
    gap_duration: float = 60.0,
    sample_rate: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    Generate a session with multiple segments (mask-off events).

    Args:
        segment_count: Number of continuous segments
        breaths_per_segment: Breaths in each segment
        gap_duration: Duration of gaps between segments in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (timestamps, flow_values, gap_markers)
        gap_markers is list of (gap_start, gap_end) timestamps
    """
    all_timestamps = []
    all_flow_values = []
    gap_markers = []
    current_time = 0.0

    for i in range(segment_count):
        # Generate segment
        t, flow = create_session(
            num_breaths=breaths_per_segment, sample_rate=sample_rate
        )

        # Offset to current time
        t = t + current_time
        current_time = t[-1]

        all_timestamps.extend(t)
        all_flow_values.extend(flow)

        # Add gap (except after last segment)
        if i < segment_count - 1:
            gap_start = current_time
            current_time += gap_duration
            gap_markers.append((gap_start, current_time))

    return (np.array(all_timestamps), np.array(all_flow_values), gap_markers)
