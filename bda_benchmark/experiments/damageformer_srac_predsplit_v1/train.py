"""Reuse the original training protocol without editing its files."""
import argparse
from functools import partial
from bda_benchmark.experiments.damageformer_srac_approx_v1 import train as original
from .modules import SplitRegionModule

VARIANTS = {
    's3_original': (True, True, True),
    's3_predsplit': (True, True, True),
    'grid_predsplit': (False, True, True),
}


def main():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--variant', required=True, choices=list(VARIANTS))
    args, _ = probe.parse_known_args()
    print('PredSplit V1: threshold=0.5; detached A1 prediction; '
          'pool AND projection split; unchanged unsplit partition loss; '
          'no Memory; no Test. Fresh head initialization.', flush=True)
    # Only this Python process sees the injected factory/variant registry.
    # CLI, optimizer, losses, seed, data order and checkpoint policy stay original.
    previous_factory, previous_variants = original.RegionModule, original.VARIANTS
    try:
        original.VARIANTS = VARIANTS
        original.RegionModule = partial(SplitRegionModule,
            split=args.variant != 's3_original')
        original.main()
    finally:
        original.RegionModule, original.VARIANTS = previous_factory, previous_variants


if __name__ == '__main__':
    main()
