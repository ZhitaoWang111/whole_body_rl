#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def reach_ee_curve(dist, max_dist, k):
    exp_max = np.exp(-k * max_dist)
    return (np.exp(-k * dist) - exp_max) / (1.0 - exp_max + 1e-6)


def main():
    parser = argparse.ArgumentParser(description="Plot reach_ee reward curve")
    parser.add_argument("--max-dist", type=float, default=0.5, help="reach_ee_max_dist")
    parser.add_argument("--k", type=float, default=4.0, help="reach_ee_exp_k")
    parser.add_argument("--num", type=int, default=400, help="number of samples")
    parser.add_argument("--out", type=str, default="", help="save figure path (optional)")
    args = parser.parse_args()

    dist = np.linspace(0.0, args.max_dist, args.num)
    reward = reach_ee_curve(dist, args.max_dist, args.k)

    plt.figure(figsize=(6, 4))
    plt.plot(dist, reward, label=f"exp(k={args.k})")
    plt.xlabel("dist_ee (m)")
    plt.ylabel("reach_ee")
    plt.title("reach_ee reward curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
