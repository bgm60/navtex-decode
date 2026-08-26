"""
Navtex_Decoder - Normalize Audio Input
======================================

Author: Brian Martlew
Date: 25 Aug 2026

This is a simple routine to normalize the input audio buffer into the range [=1.0, 1.0].
In order to prevent assymmetry, the zero point is maintained.
"""

from __future__ import annotations

import numpy as np

def bgm_normalize(arr):
    arr = np.asarray(arr, dtype=float)
    max_val = np.max(np.abs(arr))

    if max_val == 0:
        return np.zeros_like(arr)

    return arr/max_val

def _demo():
    print("Testing...")

    test_data = np.array([-10, 0, 5, 20])
    print(bgm_normalize(test_data))

    print("complete!")

if __name__ == "__main__":
    _demo()
