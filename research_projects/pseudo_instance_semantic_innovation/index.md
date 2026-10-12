# Pseudo-instance semantic innovation project

Research question: can prediction-only building structure improve the locked
P3 semantic result beyond 60.5020% without official instance training?

Current evidence:

- LocCC + majority: 61.3790% (+0.8770 pp), promising but event CI crosses zero.
- Locked LocCC+A6: approximately +1.04 pp and beats a matched local control.
- MICB broadcast: -2.9983 pp; do not reuse hard bag broadcasting.
- Component-balanced T1: 59.5607%; do not revive equal-component weighting.
- Oracle instances show a large upper bound, but are diagnostic only.
- RGID C3 versus LocCC C2 remains the required splitting experiment.

The new isolated branch tests selective soft context before any additional
training. See `bda_benchmark/experiments/spi_selective_context_v1/`.

