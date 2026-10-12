"""Print snapshot identity before the shared J1/J2/J3 progress report."""
import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_joint123_v1.report import main as joint_report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    path = Path(args.root) / "manifest.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        print("SNAPSHOT PROTOCOL, NOT the completed-50000-step parent protocol")
        for key in ("parent_checkpoint_step", "declared_parent_steps", "parent_run_complete", "front_sha256"):
            print(key + ":", data.get(key))
    joint_report()


if __name__ == "__main__":
    main()
