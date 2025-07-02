
import numpy as np
from matplotlib import pyplot as plt

def add_awgn_noise(signal, snr_db):
    """Add AWGN noise to a real-valued signal."""
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(signal**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.normal(0, 1, len(signal))
    return signal + noise


##---plot eye diagram---##
def plot_eye_diagram(signal, sps=1, num_traces=100, span=4, title="Eye Diagram"):
    """
    Plot a custom eye diagram using Matplotlib.
    :param signal: 1D signal array
    :param sps: Samples per symbol
    :param num_traces: Number of traces to overlay
    :param span: How many symbols to show per trace
    :param title: Title of the plot
    """
    samples_per_trace = sps * span
    traces = []

    for i in range(num_traces):
        start = i * sps
        end = start + samples_per_trace
        if end <= len(signal):
            traces.append(signal[start:end])

    traces = np.array(traces).T
    plt.figure(figsize=(8, 4))
    plt.plot(traces, color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def plot_constellation(signal, title="Constellation"):
    plt.figure()
    plt.scatter(np.real(signal), np.imag(signal), alpha=0.5)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
###--------###