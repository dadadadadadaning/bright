BRIGHT: original GT + six existing model predictions

This is an assembly script, NOT training or inference. No Torch, CUDA or
ChangeMamba extension is required. Dependencies: Python, numpy, Pillow.
Input predictions and dataset files are read-only.

1. Upload assemble_six_models_gt.py to:
   /root/shared-nvme/BRIGHT-gallery-tools/

2. Run in the server terminal:

conda activate /root/shared-nvme/conda-envs/bright4090
cd /root/shared-nvme/BRIGHT-gallery-tools
python -u assemble_six_models_gt.py \
  --repo-root /root/shared-nvme/BRIGHT-master \
  --data /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --predictions /root/shared-nvme/BRIGHT-Test878-gallery-v2/predictions \
  --out /root/shared-nvme/BRIGHT-sixmodels-GT-v1 \
  --zip

3. Download this server file using your existing server file manager:
   /root/shared-nvme/BRIGHT-sixmodels-GT-v1.zip

4. Extract the WHOLE folder and open index.html in a browser.
   Scroll horizontally for the 9-column strip. Click the figure for full size.
   Filter by disaster or search event/sample ID; check cases and export IDs.

Columns:
Pre-event RGB | Post-event SAR | GT | UNet | DeepLabV3+ |
SiamAttnUNet | SiamCRNN | DamageFormer | A1

GT is read from target/{sample_id}_building_damage.tif. It is NOT predicted.
GT exports: ground_truth/{hazard}/{event}/*_GT_color.png and *_GT_labels.png.
The exported raw label PNG is checked against the source pixel-for-pixel.
Original IDs: 0 background, 1 intact, 2 damaged, 3 destroyed, 255 ignore.
Dataset label correctness itself is not independently adjudicated.
Predictions must have the same spatial dimensions and IDs 0..3.
Missing predictions cause failure instead of invented/blank results.
Full field of view is retained. Categorical display uses nearest-neighbor.
SAR percentile stretch is display-only. A1 is the existing recorded model,
not a claim that the proposed region-memory architecture has been evaluated.

The default run uses ALL Test IDs, without filtering for performance.
Output is grouped by disaster type/event. No metrics are recomputed.
ZIP contains visual comparisons, original GT exports, provenance and index,
but no checkpoints. ZIP creation adds an extra copy of the output on disk.
Omit --zip if disk space is tight and download the output folder instead.
Use a NEW --out directory on retry; nonempty output directories are rejected
to avoid mixing runs or subsets. Do not delete the original prediction cache.

For selected cases, add --ids /path/to/selected_test_ids.txt and use a new
output directory. This produces a qualitative subset, not full Test metrics.
