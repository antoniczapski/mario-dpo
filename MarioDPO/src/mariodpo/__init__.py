"""MarioDPO — preference-aligned Mario level generation.

A clean re-implementation of the MarioDPO experiment:
  * static-feature Judge Function (no generator-identity leakage),
  * MarioGPT continue-pretraining into the arena 16x37 representation,
  * DPO alignment on oversampled human votes + judge-labelled synthetic pairs.

See README.md for the full pipeline and run instructions.
"""

__version__ = "0.1.0"
