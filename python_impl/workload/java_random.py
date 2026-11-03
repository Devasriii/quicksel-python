"""An exact Python copy of ``java.util.Random``.

Why do we need this?  The Java workload generators (SyntheticDataGenerator.java)
create query rectangles with ``new Random(0)``.  numpy's random numbers are
different, so to get *the very same* rectangles as the original code we
re-implement Java's generator.  It is a tiny "linear congruential generator"
(LCG): every step computes  seed = (seed * 0x5DEECE66D + 0xB) mod 2^48.

Only the methods the original code uses are implemented.
"""

from __future__ import annotations

_MULTIPLIER = 0x5DEECE66D
_ADDEND = 0xB
_MASK = (1 << 48) - 1  # keep 48 bits


class JavaRandom:
    """Behaves exactly like ``new java.util.Random(seed)``."""

    def __init__(self, seed: int = 0) -> None:
        # Java "scrambles" the seed before using it.
        self._seed = (seed ^ _MULTIPLIER) & _MASK

    def _next(self, bits: int) -> int:
        """Java's protected ``next(bits)``: advance the LCG, return the top `bits` bits."""
        self._seed = (self._seed * _MULTIPLIER + _ADDEND) & _MASK
        result = self._seed >> (48 - bits)
        # Java returns a signed 32-bit int; convert if the top bit is set.
        if result >= 1 << 31:
            result -= 1 << 32
        return result

    def next_double(self) -> float:
        """Java's ``nextDouble()``: uniform float in [0, 1) built from 53 random bits."""
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))
