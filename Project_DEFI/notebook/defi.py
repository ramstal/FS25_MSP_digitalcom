import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ------------------------
# Parameters
# ------------------------
num_symbols = 1000
SNR_dB = 10
mu = 0.001
L1, L2 = 7, 5                 # DFE tap lengths
train_len = 900              # Training length

# ------------------------
# Generate BPSK symbols
# ------------------------
bits = np.random.randint(0, 2, num_symbols)
symbols = 2 * bits - 1

# ------------------------
# ISI Channel (3-tap)
# ------------------------
h = np.array([0.5, 0.8, 0.3])
tx_signal = lfilter(h, 1, symbols)

# ------------------------
# Add AWGN
# ------------------------
signal_power = np.mean(tx_signal**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise
rx_signal = rx_signal[:num_symbols]  # align

# ------------------------
# BER before equalization
# ------------------------
decisions_before = np.where(rx_signal >= 0, 1, -1)
ber_before = np.mean(decisions_before != symbols)
print(f"BER before equalization: {ber_before:.4f}")

# ------------------------
# Eye Diagram Function
# ------------------------
def plot_eye(signal, span=2, sps=1, traces=100, title="Eye Diagram"):
    per_trace = span * sps
    plt.figure(figsize=(8, 3))
    for i in range(traces):
        start = i * sps
        end = start + per_trace
        if end < len(signal):
            plt.plot(signal[start:end], alpha=0.3, color='blue')
    plt.title(title)
    plt.grid(True)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

# Uncomment to view:
# plot_eye(rx_signal, title="Eye Diagram Before Equalization")

# ------------------------
# Linear LMS Equalizer
# ------------------------
def linear_lms_equalizer(rx, tx, L=9, mu=0.0005, Ntrain=500):
    N = len(rx)
    taps = np.zeros(L)
    taps[L//2] = 1.0  # center tap
    y = np.zeros(N)
    decisions = np.zeros(N)
    padded_rx = np.concatenate((rx, [0]*(L-1)))
    for n in range(N):
        x_in = padded_rx[n:n+L][::-1]
        y[n] = np.dot(taps, x_in)
        decisions[n] = 1 if y[n] >= 0 else -1
        if n < Ntrain:
            e = tx[n] - y[n]
            taps += mu * e * x_in
    return y, decisions

# ------------------------
# Run Linear Equalizer
# ------------------------
y_lin, dec_lin = linear_lms_equalizer(rx_signal, symbols)
ber_lin = np.mean(dec_lin != symbols)
print(f"BER after linear equalizer: {ber_lin:.4f}")

# ------------------------
# Decision Feedback Equalizer (DFE)
# ------------------------
def decision_feedback_equalizer(rx, tx, L1=7, L2=5, mu=0.0005, Ntrain=900):
    N = len(rx)
    ff = np.zeros(L1)
    fb = np.zeros(L2)
    ff[L1//2] = 1.0  # center tap

    y = np.zeros(N)
    decisions = np.zeros(N)
    buffer = np.zeros(max(L1, L2))

    for n in range(N):
        buffer = np.concatenate(([rx[n]], buffer))[:max(L1, L2)]
        ff_in = buffer[:L1][::-1]
        fb_in = decisions[max(0, n - L2):n][::-1]
        fb_in = np.pad(fb_in, (L2 - len(fb_in), 0))

        y[n] = np.dot(ff, ff_in) - np.dot(fb, fb_in)
        decisions[n] = 1 if y[n] >= 0 else -1

        if n < Ntrain:
            e = tx[n] - y[n]
            ff += mu * e * ff_in
            fb += mu * e * fb_in

    return y, decisions

# ------------------------
# Run DFE
# ------------------------
y_dfe, dec_dfe = decision_feedback_equalizer(rx_signal, symbols, L1, L2, mu, train_len)
ber_dfe = np.mean(dec_dfe != symbols)
print(f"BER after DFE: {ber_dfe:.4f}")

# Uncomment to view:
# plot_eye(y_dfe, title="Eye Diagram After DFE")
