# External Optical–SAR CD → BRIGHT protocol

This experiment separates a runnable controlled baseline from faithful
third-party reproduction.

- `x1`: repository-native plain non-shared RGB/SAR encoders, simple
  concat/absolute-difference fusion and one four-class semantic head.
- `x2`: official HeteCD Transformer encoders and 3D difference decoder,
  adapted only at the SAR input (1 channel) and BRIGHT output (4 classes).

HeteCD is research-only software. Its source is not copied into this project;
place the authors' repository at `third_party/HeteCD` and cite the original
paper/repository. Do not label `x1` as HeteCD.

The controlled first round uses one training objective:

`CE(BG/Intact/Damaged/Destroyed) + 0.75 * Lovasz`

No separate localisation, SAR auxiliary, hierarchy, routing, ordinal or
instance loss is enabled. Localisation metrics are derived from the four-class
prediction (`background` versus `logsumexp(building classes)`). Optional
`FCA_WEIGHT` is reserved for the BRIGHT-safe unchanged-region consistency
ablation after the no-FCA x2 control has run.

Run order:

1. `x1` smoke and 1500-step screen.
2. Obtain the official HeteCD source and run `x2` with `FCA_WEIGHT=0`.
3. Only then run the controlled FCA ablation with one frozen weight.
4. Do not access test until the model and hyperparameters are locked on val.
