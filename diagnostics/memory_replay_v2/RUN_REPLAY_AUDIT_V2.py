from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

EVIDENCE_SHA = "d05ca254bbdff51e0c7cc08c6dbce1aec1fac8b55a4dadbd8f6d0f7506306213"
SCRIPT_SHA = "5c36a67f41c9c671fb1bd887d6031f2a4877ac1c788e0138195bb364a9064545"


def find_evidence(base):
    matches = []
    for summary in base.glob("*/keys_smoke/summary.json"):
        try:
            record = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        saved = record.get("last_replay_evidence") or {}
        if saved.get("sha256") != EVIDENCE_SHA:
            continue
        evidence = summary.parent / saved["path"]
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != EVIDENCE_SHA:
            raise SystemExit("失败证据的实际校验和不一致，保留文件并停止。")
        matches.append((summary.parent.parent, evidence))
    if len(matches) != 1:
        raise SystemExit(f"找到 {len(matches)} 个匹配目录，无法唯一定位：{matches}")
    return matches[0]


def main():
    root, evidence = find_evidence(Path("/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_KEY_COMPARE_V2"))
    script = Path("/root/shared-nvme/AUDIT_MEMORY_REPLAY_V2.py")
    if not script.exists():
        url = "https://github.com/dadadadadadaning/bright/raw/fb136104b1a835dee0996cc8f9ca66b30fcca12f/diagnostics/memory_replay_v2/AUDIT_MEMORY_REPLAY_V2.py"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SCRIPT_SHA:
            raise SystemExit("下载脚本校验失败，停止。")
        with script.open("xb") as handle:
            handle.write(data)
    if hashlib.sha256(script.read_bytes()).hexdigest() != SCRIPT_SHA:
        raise SystemExit("已有探针脚本校验不一致；保留文件并停止。")
    output = root / ("replay_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    print("已定位失败目录：", root, "\n诊断输出：", output, flush=True)
    args = [sys.executable, str(script), "--protocol",
            "/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json",
            "--evidence", str(evidence), "--evidence-sha256", EVIDENCE_SHA, "--output", str(output)]
    result = subprocess.run(args)
    print("诊断记录：", output / "summary.json", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
