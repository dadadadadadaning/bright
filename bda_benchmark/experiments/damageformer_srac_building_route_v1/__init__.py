"""Matched building/background routing on a frozen SRAC Q.

The original 9-neighbour Q is never changed by this package.  A routed
readout may use it directly, or split each seed into predicted background and
building slots.  All routing inputs are available at inference time.
"""
