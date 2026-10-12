# -*- coding: utf-8 -*-
from pathlib import Path

src = Path("script/standard_ML/train_DamageFormer_BGMP_accum.py")
dst = Path("script/standard_ML/train_DamageFormer_BGMP_K3Gated_accum.py")

if not src.exists():
    raise FileNotFoundError(src)

text = src.read_text(encoding="utf-8")

old_import = "from model.DamageFormer_BGMP import DamageFormer_BGMP"
new_import = (
    "from model.DamageFormer_BGMP_K3Gated "
    "import DamageFormer_BGMP_K3Gated"
)
if old_import not in text:
    raise RuntimeError("Old BGMP import was not found.")
text = text.replace(old_import, new_import)

# Update only the diagnostic type annotation; do not globally replace names.
text = text.replace(
    "model: DamageFormer_BGMP,",
    "model: DamageFormer_BGMP_K3Gated,",
)

text = text.replace(
    'p.add_argument("--proto_k", type=int, default=1)',
    'p.add_argument("--proto_k", type=int, default=3, '
    'help="Compatibility option; S3V5 uses fixed [1,3,3].")',
)
text = text.replace(
    'p.add_argument("--proto_dim", type=int, default=128)',
    'p.add_argument("--proto_dim", type=int, default=256)',
)
text = text.replace(
    'p.add_argument("--model_type", type=str, default="S2_BGMP_K1")',
    'p.add_argument("--model_type", type=str, '
    'default="S3V5_K133_HardGatedLogit")',
)

marker = '    p.add_argument("--learning_rate", type=float, default=1e-4)\n'
extra_args = """    p.add_argument(
        "--core_confidence",
        type=float,
        default=0.70,
    )
    p.add_argument(
        "--lambda_damaged",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--lambda_destroyed",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--proto_evidence_clip",
        type=float,
        default=4.0,
    )

"""
if "--core_confidence" not in text:
    if marker not in text:
        raise RuntimeError("Learning-rate argument marker was not found.")
    text = text.replace(marker, extra_args + marker)

old_model = """    model = DamageFormer_BGMP(
        roi_alpha=args.roi_alpha,
        proto_k=args.proto_k,
        proto_dim=args.proto_dim,
        proto_momentum=proto_micro_momentum,
        proto_tau=args.proto_tau,
        proto_warmup_iters=args.proto_warmup_iters,
    ).to(device)
"""

new_model = """    model = DamageFormer_BGMP_K3Gated(
        roi_alpha=args.roi_alpha,
        proto_dim=args.proto_dim,
        proto_momentum=proto_micro_momentum,
        proto_tau=args.proto_tau,
        proto_warmup_iters=args.proto_warmup_iters,
        core_confidence=args.core_confidence,
        lambda_damaged=args.lambda_damaged,
        lambda_destroyed=args.lambda_destroyed,
        evidence_clip=args.proto_evidence_clip,
    ).to(device)
"""

if old_model not in text:
    raise RuntimeError(
        "Model-construction block was not found. "
        "Your local trainer may differ from the expected version."
    )
text = text.replace(old_model, new_model)

text = text.replace(
    'print(f"  proto K / dim / tau    : '
    '{args.proto_k} / {args.proto_dim} / {args.proto_tau}", flush=True)',
    'print(f"  proto K roles / dim/tau: '
    '[1,3,3] / {args.proto_dim} / {args.proto_tau}", flush=True)',
)

dst.write_text(text, encoding="utf-8")
print("WROTE:", dst)
