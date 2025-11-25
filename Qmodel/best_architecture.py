import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_reward_from_features(features):
    """
    features = [MainLobe, SSL, HPBW, Theta0]
    Must match the reward definition used in AntennaEnv.step().
    """
    MainLobe, SSL, HPBW, Theta0 = features
    reward = MainLobe - abs(SSL) - HPBW
    return float(reward)


def main():
    # Resolve paths relative to this file, not to the current working directory
    this_dir = Path(__file__).resolve().parent
    dataset_dir = this_dir.parent / "Dataset"

    minput_path = dataset_dir / "Minput.npy"
    moutput_path = dataset_dir / "Moutput.npy"

    print(f"Loading Minput from:  {minput_path}")
    print(f"Loading Moutput from: {moutput_path}")

    Minput = np.load(minput_path)   # shape (4, N)
    Moutput = np.load(moutput_path) # shape (5, N)

    num_samples = Minput.shape[1]
    print(f"Number of architectures in dataset: {num_samples}")

    best_idx = None
    best_reward = -1e9

    # Scan all architectures and compute reward
    for i in range(num_samples):
        features = Minput[:, i]   # [MainLobe, SSL, HPBW, Theta0]
        r = compute_reward_from_features(features)
        if r > best_reward:
            best_reward = r
            best_idx = i

    # Safety check
    if best_idx is None:
        print("ERROR: No best architecture found (num_samples == 0 ?)")
        return

    # Extract best features and architecture
    best_features = Minput[:, best_idx]
    best_arch = Moutput[:, best_idx].astype(int)

    # --- Print info in console ---
    print("\n=== Best Architecture according to the environment reward ===")
    print(f"Index in dataset: {best_idx}")
    print(f"Best reward: {best_reward:.3f}\n")

    print("RF performance (Minput[:, idx]):")
    print(f"  MainLobe : {best_features[0]:.3f}")
    print(f"  SSL      : {best_features[1]:.3f}")
    print(f"  HPBW     : {best_features[2]:.3f}")
    print(f"  Theta0   : {best_features[3]:.3f}\n")

    print("Antenna architecture (Moutput[:, idx]):")
    for ring_id, elems in enumerate(best_arch, start=1):
        print(f"  Ring {ring_id}: {elems} elements")

    # --- Plot the best architecture as a bar chart ---
    rings = np.arange(1, len(best_arch) + 1)

    plt.figure(figsize=(6, 4))
    plt.bar(rings, best_arch)
    plt.xticks(rings, [f"Ring {i}" for i in rings])
    plt.xlabel("Ring index")
    plt.ylabel("Number of elements")
    plt.title(f"Best architecture (idx={best_idx}, reward={best_reward:.2f})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = this_dir / "best_architecture.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved plot to: {out_path}")

    # Try to show the plot (may do nothing in some environments, but file is saved)
    try:
        plt.show()
    except Exception as e:
        print(f"plt.show() failed with: {e}")


if __name__ == "__main__":
    main()
