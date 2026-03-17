import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Signal parameters
freq = 5       # Hz
fs = 1000      # sample rate
duration = 4 / freq  # exactly 4 cycles
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
N = len(t)

# Generate waves
sine     = np.sin(2 * np.pi * freq * t)
square   = signal.square(2 * np.pi * freq * t)
sawtooth = signal.sawtooth(2 * np.pi * freq * t)

waves = [
    ("Sine",     sine,     "tab:blue"),
    ("Square",   square,   "tab:orange"),
    ("Sawtooth", sawtooth, "tab:green"),
]

fig, axes = plt.subplots(3, 2, figsize=(13, 8))
fig.suptitle("Time Domain (4 cycles) vs Frequency Domain (FFT)", fontsize=14)

for row, (name, w, color) in enumerate(waves):
    # --- Time domain ---
    ax_t = axes[row, 0]
    ax_t.plot(t, w, color=color)
    ax_t.set_title(f"{name} — Time Domain")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude")
    ax_t.set_xlim(t[0], t[-1])
    ax_t.grid(True, alpha=0.3)

    # --- Frequency domain (FFT) ---
    ax_f = axes[row, 1]
    freqs = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(np.fft.rfft(w)) / N * 2  # single-sided, normalised
    ax_f.stem(freqs, magnitude, linefmt=color, markerfmt=f"C{row}o",
              basefmt="k-")
    ax_f.set_title(f"{name} — Frequency Domain")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("|Amplitude|")
    ax_f.set_xlim(0, freq * 15)   # show up to the 15th harmonic
    ax_f.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
