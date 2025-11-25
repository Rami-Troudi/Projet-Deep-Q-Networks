# plot_best_pattern.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------
# Array Factor utilities
# -----------------------

def compute_AF_pattern(elements_per_ring, theta0_deg):
    """
    Compute AF(θ) in dB for a circular multi-ring array,
    with the same physical assumptions as the TP.

    elements_per_ring : array-like of length 5 (N1..N5)
    theta0_deg        : steering angle in degrees (from Minput[3, idx])
    """
    c = 3e8
    f = 2.45e9
    wavelength = c / f
    k = 2 * np.pi / wavelength

    r0 = 0.2 * wavelength
    delta_r = 0.5 * wavelength

    elements_per_ring = np.array(elements_per_ring, dtype=int)
    max_rings = len(elements_per_ring)

    # Angular grid
    theta = np.linspace(0, 2 * np.pi, 1000)
    theta0 = np.deg2rad(theta0_deg)

    AF = np.zeros_like(theta, dtype=complex)

    for ring_idx in range(max_rings):
        N = elements_per_ring[ring_idx]
        if N <= 0:
            continue

        r = r0 + ring_idx * delta_r

        for m in range(N):
            phi_m = 2 * np.pi * m / N
            # Simplified version of the phase term used in the TP
            phase = k * r * np.cos(phi_m) * (np.sin(theta) - np.sin(theta0))
            AF += np.exp(1j * phase)

    # Normalize and convert to dB
    AF_abs = np.abs(AF)
    AF_abs /= (AF_abs.max() + 1e-12)
    AF_dB = 20 * np.log10(AF_abs + 1e-12)
    AF_dB[AF_dB < -40] = -40.0

    theta_deg = np.rad2deg(theta)
    return theta, theta_deg, AF_dB


# -----------------------
# Find best architecture (same reward as env_dataset)
# -----------------------

def find_best_index(Minput):
    """
    Returns index that maximizes the reward used in AntennaEnv:
    reward = ml_norm - ssl_norm - hpbw_norm
    """
    Minput_T = Minput.T  # (N, 4)
    ml = Minput_T[:, 0]
    ssl = Minput_T[:, 1]
    hpbw = Minput_T[:, 2]

    ml_min, ml_max = ml.min(), ml.max()
    ssl_min, ssl_max = ssl.min(), ssl.max()
    hpbw_min, hpbw_max = hpbw.min(), hpbw.max()

    def reward_from_features(features):
        MainLobe, SSL, HPBW, Theta0 = features
        ml_norm = (MainLobe - ml_min) / (ml_max - ml_min + 1e-8)
        ssl_norm = (SSL - ssl_min) / (ssl_max - ssl_min + 1e-8)
        hpbw_norm = (HPBW - hpbw_min) / (hpbw_max - hpbw_min + 1e-8)
        return ml_norm - ssl_norm - hpbw_norm

    rewards = np.array([reward_from_features(f) for f in Minput_T])
    best_idx = int(np.argmax(rewards))
    best_reward = float(rewards[best_idx])
    return best_idx, best_reward


def main():
    this_dir = Path(__file__).resolve().parent
    dataset_dir = this_dir.parent / "Dataset"

    Minput = np.load(dataset_dir / "Minput.npy")   # (4, N)
    Moutput = np.load(dataset_dir / "Moutput.npy") # (5, N)

    # Find best index according to the same reward as the env
    best_idx, best_reward = find_best_index(Minput)

    # Extract architecture + steering angle
    elements_per_ring = Moutput[:, best_idx].astype(int)
    features = Minput[:, best_idx]
    theta0_deg = float(features[3])

    print(f"Best index    : {best_idx}")
    print(f"Best reward   : {best_reward:.3f}")
    print(f"Theta0 (deg)  : {theta0_deg:.2f}")
    print(f"Elements/ring : {elements_per_ring}")

    # Compute AF pattern
    theta, theta_deg, AF_dB = compute_AF_pattern(elements_per_ring, theta0_deg)

    # Polar plot
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, AF_dB)
    ax.set_title(
        f"Radiation pattern (idx={best_idx}, reward={best_reward:.2f})",
        va="bottom"
    )
    ax.set_rlim(-40, 0)

 

    out_polar = this_dir / "best_pattern_polar.png"
    plt.savefig(out_polar, dpi=300)  # last figure = cartesian

    print(f"Saved polar plot in: {this_dir}")
    plt.show()


if __name__ == "__main__":
    main()
