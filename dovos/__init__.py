"""Dual-oracle protocol for annotation-free video object segmentation.

Modules: data (loading), detector (DETR objectness), propagation (SAM 2), commit (ground-truth-free
track commitment), qrd (quartile re-detection), oracles (seeding rules O1-O4), metrics and evaluate
(J&F under greedy, bijective and non-overlapping matching).
"""
__version__ = "1.0.0"
