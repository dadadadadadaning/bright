from .decoder_factory import HierarchicalChangeDecoder


class ChangeDecoder(HierarchicalChangeDecoder):
    """
    B1: Multi-relation ChangeDecoder for BRIGHT.

    Compared with original B0:
      B0 uses only:       ("cat",)
      B1 uses stage-wise: ("cat", "interleave", "split")

    For stability and memory control:
      - The first 3 low-resolution stages use cat + interleave + split.
      - The last high-resolution stage keeps cat only.
    """

    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__(
            encoder_dims=encoder_dims,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            fusion_modes_by_stage=[
                ("cat", "interleave", "split"),
                ("cat", "interleave", "split"),
                ("cat", "interleave", "split"),
                ("cat",),
            ],
            use_vss_by_stage=[True, True, True, False],
            **kwargs,
        )
