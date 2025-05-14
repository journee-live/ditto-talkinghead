"""
Debug utilities for Ditto animation system.
Provides tools for visualizing and debugging animation data.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_keypoint_sequence(
    original_seq: np.ndarray,
    smoothed_seq: np.ndarray,
    indices: list[int] = None,
    title: str = "Keypoint Sequence Comparison",
    output_path: str = None,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 150,
) -> Figure:
    """
    Plot keypoint sequences before and after smoothing and save to a PNG file.

    Args:
        original_seq: Original keypoint sequence with shape (batch, frames, dims)
        smoothed_seq: Smoothed keypoint sequence with shape (batch, frames, dims)
        indices: List of keypoint indices to plot. If None, plots the first 6 indices.
        title: Title for the plot
        output_path: Path to save the plot. If None, doesn't save.
        figsize: Figure size (width, height) in inches
        dpi: DPI for the output image

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Default to first batch item
    original = original_seq[0]
    smoothed = smoothed_seq[0]

    # Get number of frames
    n_frames = original.shape[0]
    time_axis = np.arange(n_frames)

    # Default to first 6 indices if not specified
    if indices is None:
        # Head movement parameters are typically in the first few indices
        indices = list(range(6))

    # Create figure
    fig, axes = plt.subplots(len(indices), 1, figsize=figsize, sharex=True)
    if len(indices) == 1:
        axes = [axes]

    # Plot each keypoint index
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(time_axis, original[:, idx], "b-", label="Original", alpha=0.7)
        ax.plot(time_axis, smoothed[:, idx], "r-", label="Smoothed", alpha=0.7)
        ax.set_ylabel(f"KP {idx}")
        ax.grid(True, alpha=0.3)

        # Only show legend on first plot
        if i == 0:
            ax.legend()

    # Set common labels
    axes[-1].set_xlabel("Frames")
    fig.suptitle(title)
    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to {output_path}")

    return fig


def plot_keypoint_frequency_analysis(
    original_seq: np.ndarray,
    smoothed_seq: np.ndarray,
    indices: list[int] = None,
    title: str = "Keypoint Frequency Analysis",
    output_path: str = None,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 150,
) -> Figure:
    """
    Plot frequency analysis of keypoint sequences before and after smoothing.

    Args:
        original_seq: Original keypoint sequence with shape (batch, frames, dims)
        smoothed_seq: Smoothed keypoint sequence with shape (batch, frames, dims)
        indices: List of keypoint indices to analyze. If None, analyzes the first 6 indices.
        title: Title for the plot
        output_path: Path to save the plot. If None, doesn't save.
        figsize: Figure size (width, height) in inches
        dpi: DPI for the output image

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Default to first batch item
    original = original_seq[0]
    smoothed = smoothed_seq[0]

    # Default to first 6 indices if not specified
    if indices is None:
        indices = list(range(6))

    # Create figure
    fig, axes = plt.subplots(len(indices), 2, figsize=figsize)
    if len(indices) == 1:
        axes = [axes]

    # Plot each keypoint index
    for i, idx in enumerate(indices):
        # Time domain plot
        ax_time = axes[i][0]
        ax_time.plot(original[:, idx], "b-", label="Original", alpha=0.7)
        ax_time.plot(smoothed[:, idx], "r-", label="Smoothed", alpha=0.7)
        ax_time.set_ylabel(f"KP {idx}")
        ax_time.set_xlabel("Frames")
        ax_time.grid(True, alpha=0.3)
        if i == 0:
            ax_time.legend()

        # Frequency domain plot
        ax_freq = axes[i][1]

        # Compute FFT for original and smoothed
        orig_fft = np.abs(np.fft.rfft(original[:, idx]))
        smooth_fft = np.abs(np.fft.rfft(smoothed[:, idx]))
        freqs = np.fft.rfftfreq(len(original[:, idx]))

        ax_freq.plot(freqs, orig_fft, "b-", label="Original", alpha=0.7)
        ax_freq.plot(freqs, smooth_fft, "r-", label="Smoothed", alpha=0.7)
        ax_freq.set_xlabel("Frequency")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.grid(True, alpha=0.3)
        if i == 0:
            ax_freq.legend()

    fig.suptitle(title)
    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        print(f"Frequency analysis plot saved to {output_path}")

    return fig


def capture_smoothing_comparison(
    res_kp_seq: np.ndarray,
    smoothing_function,
    smoothing_params: dict[str, Any],
    plot_indices: list[int] = None,
    output_dir: str = "./debug_plots",
    prefix: str = "smoothing_comparison",
) -> tuple[str, str]:
    """
    Capture and save plots comparing keypoint sequences before and after smoothing.

    Args:
        res_kp_seq: Original keypoint sequence with shape (batch, frames, dims)
        smoothing_function: Function that performs smoothing, should take res_kp_seq and other params
        smoothing_params: Dictionary of parameters to pass to the smoothing function
        plot_indices: List of keypoint indices to plot. If None, plots default indices.
        output_dir: Directory to save plots
        prefix: Prefix for output filenames

    Returns:
        Tuple[str, str]: Paths to the saved time domain and frequency domain plots
    """
    # Make a copy of the original sequence
    original_seq = res_kp_seq.copy()

    # Apply smoothing
    smoothed_seq = smoothing_function(original_seq.copy(), **smoothing_params)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames
    time_plot_path = os.path.join(output_dir, f"{prefix}_time_domain.png")
    freq_plot_path = os.path.join(output_dir, f"{prefix}_frequency_domain.png")

    # Create plots
    plot_keypoint_sequence(
        original_seq,
        smoothed_seq,
        indices=plot_indices,
        title="Keypoint Sequence Before and After Smoothing",
        output_path=time_plot_path,
    )

    plot_keypoint_frequency_analysis(
        original_seq,
        smoothed_seq,
        indices=plot_indices,
        title="Frequency Analysis Before and After Smoothing",
        output_path=freq_plot_path,
    )

    return time_plot_path, freq_plot_path


def plot_smoothing_parameter_comparison(
    res_kp_seq: np.ndarray,
    smoothing_function,
    param_name: str,
    param_values: list[Any],
    fixed_params: dict[str, Any],
    plot_indices: list[int] = None,
    output_path: str = None,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 150,
) -> Figure:
    """
    Plot comparison of different smoothing parameter values.

    Args:
        res_kp_seq: Original keypoint sequence with shape (batch, frames, dims)
        smoothing_function: Function that performs smoothing
        param_name: Name of the parameter to vary
        param_values: List of values for the parameter to compare
        fixed_params: Dictionary of fixed parameters for the smoothing function
        plot_indices: List of keypoint indices to plot. If None, plots the first index.
        output_path: Path to save the plot. If None, doesn't save.
        figsize: Figure size (width, height) in inches
        dpi: DPI for the output image

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Default to first index if not specified
    if plot_indices is None:
        plot_indices = [0]

    # Create figure with subplots for each index
    fig, axes = plt.subplots(len(plot_indices), 1, figsize=figsize, sharex=True)
    if len(plot_indices) == 1:
        axes = [axes]

    # Get original sequence (first batch)
    original = res_kp_seq[0]
    n_frames = original.shape[0]
    time_axis = np.arange(n_frames)

    # Plot original sequence and smoothed sequences with different parameter values
    for i, idx in enumerate(plot_indices):
        ax = axes[i]

        # Plot original
        ax.plot(time_axis, original[:, idx], "k-", label="Original", alpha=0.5)

        # Plot smoothed with different parameter values
        for value in param_values:
            # Create params dictionary with current value
            params = fixed_params.copy()
            params[param_name] = value

            # Apply smoothing
            smoothed_seq = smoothing_function(res_kp_seq.copy(), **params)
            smoothed = smoothed_seq[0]

            # Plot smoothed
            ax.plot(
                time_axis, smoothed[:, idx], "-", label=f"{param_name}={value}", alpha=0.7
            )

        ax.set_ylabel(f"KP {idx}")
        ax.grid(True, alpha=0.3)

        # Only show legend on first plot
        if i == 0:
            ax.legend()

    # Set common labels
    axes[-1].set_xlabel("Frames")
    fig.suptitle(f"Effect of {param_name} on Smoothing")
    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        print(f"Parameter comparison plot saved to {output_path}")

    return fig
