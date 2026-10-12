# Structure beyond splitting: paper and code search

Search date: 2026-09-04. Sources are official paper pages, proceedings, arXiv,
or author-maintained code repositories.

1. **PointRend: Image Segmentation as Rendering** (CVPR 2020). Adaptively
   computes labels only at uncertain points. Transfer: update only uncertain
   pixels instead of broadcasting one region label everywhere.
   Paper: https://arxiv.org/abs/1912.08193
   Code: https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
2. **SegFix: Model-Agnostic Boundary Refinement for Segmentation** (ECCV 2020).
   Transfers reliable interior evidence toward unreliable boundaries. Transfer:
   explicitly distinguish building core and boundary instead of treating both
   as equally trustworthy.
   Paper: https://arxiv.org/abs/2007.04269
   Code: https://github.com/openseg-group/openseg.pytorch
3. **Object-Contextual Representations for Semantic Segmentation** (ECCV 2020).
   Uses soft object-region representations and pixel-region relations. Transfer:
   blend a foreground region distribution with pixel probabilities.
   Paper: https://arxiv.org/abs/1909.11065
   Code: https://github.com/openseg-group/openseg.pytorch
4. **Bootstrapping Semantic Segmentation with Regional Contrast** (ICLR 2022).
   Applies regional contrast to sparse hard pixels with limited memory overhead.
   Transfer: if inference-only soft context passes, use sparse region-prototype
   contrast rather than the failed component-balanced loss.
   Paper: https://arxiv.org/abs/2104.04465
   Code: https://github.com/lorenmt/reco
5. **Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels**
   (CVPR 2022). Separates reliable and unreliable pseudo-labels; unreliable
   predictions are not used as ordinary positive targets. Transfer: confidence
   gates and negative evidence, never unconditional pseudo-label supervision.
   Paper: https://haochen-wang409.github.io/U2PL/resources/U2PL_CVPR_2022.pdf
   Code: https://github.com/Haochen-Wang409/U2PL
6. **The Lovasz-Softmax Loss** (CVPR 2018). Direct mIoU surrogate. Transfer:
   use only as a training baseline if a trainable module is justified; it is not
   a standalone paper innovation here.
   Paper: https://arxiv.org/abs/1705.08790
   Code: https://github.com/bermanmaxim/LovaszSoftmax
7. **Distributionally Robust Neural Networks for Group Shifts** (ICLR 2020).
   Optimizes worst-group loss with strong regularization. Transfer: events can
   be groups in a later robustness ablation, after the core module works.
   Paper: https://arxiv.org/abs/1911.08731
   Code: https://github.com/kohpangwei/group_DRO

