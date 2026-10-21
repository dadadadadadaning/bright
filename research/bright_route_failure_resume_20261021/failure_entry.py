"""Continue from verified checkpoint1500 with original route-predicate evidence."""
import argparse
import sys

import training_entry
from failure_bindings import observe_bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("resume","worker","logs","status","check-evaluation"))
    args, remaining = parser.parse_known_args()
    training_entry.observe_bindings = observe_bindings
    # Keep the original launcher, checkpoint restoration, objective, updates,
    # full boundary evaluation, final integrity checks and promotion rules.
    return training_entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
