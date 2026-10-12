# Image protocol V2

Separate train_image_protocol.py and run_image_protocol.sh; old trainer untouched.
Use fresh torchvision Swin-T pretraining, not the unstable T0 or old A1 task weights.
Formal: AdamW lr1e-4 WD5e-3, loc CE+Lovasz plus damage CE+Lovasz,
50000 actual optimizer updates, single-GPU batch4 x accumulation4 = EBS16,
200000 microbatches, 800000 training crops. Existing random crop/flips/90-degree
rotation retained, crop512. Larger microbatch is not guaranteed to cure instability.

Image's secondary table describes 8000 updates/128000 crops, not the chosen main
50000-update budget. No DDP supported; do not multiply this launch across GPUs.
Warmup1000+cosine is an explicitly retained choice because image does not specify
a scheduler; SCHEDULE=constant uses fixed1e-4 instead. Freeze this choice across
all comparisons. Do not silently change batch/crop/schedule mid-run.

Existence head absent: exist_loss=null, not a fake zero. JSONL records total/loc/cls
CE and Lovasz separately, LR, crops and microsteps. Validation also records loc
confusion/F1, damage confusion and per-event scores. best/last weights only.
No resume support. New output root required. No Test access.

First preflight, then smoke t0 and smoke t3 at batch4 to check CUDA memory.
Smoke20 still runs full Val450. Formal should begin with T0 only. Do not launch
the full sequence until baseline stability is established. If OOM, do not silently
revert to batch1; choose and document a revised common microbatch protocol.
