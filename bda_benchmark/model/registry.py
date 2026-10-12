"""Lazy registry for complete BRIGHT task models.

This module is intentionally dependency-light.  It can be imported to inspect
the model catalogue even when the deep-learning runtime is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, Mapping, Tuple, Type


@dataclass(frozen=True)
class ModelSpec:
    """Metadata needed to locate and construct one complete task model."""

    name: str
    module: str
    class_name: str
    family: str
    input_mode: str
    output_contract: str
    aliases: Tuple[str, ...] = ()
    required_kwargs: Tuple[str, ...] = ()
    default_kwargs: Tuple[Tuple[str, Any], ...] = ()
    offline_kwargs: Tuple[Tuple[str, Any], ...] = ()
    notes: str = ""

    def load_class(self) -> Type[Any]:
        """Import and return the registered class without constructing it."""

        package = __package__ or "model"
        module = import_module(self.module, package=package)
        try:
            return getattr(module, self.class_name)
        except AttributeError as exc:
            raise ImportError(
                f"{self.module!r} does not export {self.class_name!r}"
            ) from exc

    def constructor_defaults(self) -> Dict[str, Any]:
        return dict(self.default_kwargs)

    def offline_defaults(self) -> Dict[str, Any]:
        return dict(self.offline_kwargs)


def _spec(
    name: str,
    module: str,
    class_name: str,
    family: str,
    input_mode: str = "paired",
    output_contract: str = "loc_damage_tuple",
    *,
    aliases: Iterable[str] = (),
    required_kwargs: Iterable[str] = (),
    default_kwargs: Mapping[str, Any] | None = None,
    offline: bool = False,
    notes: str = "",
) -> ModelSpec:
    offline_kwargs = (("use_pretrained_swin", False),) if offline else ()
    return ModelSpec(
        name=name,
        module=module,
        class_name=class_name,
        family=family,
        input_mode=input_mode,
        output_contract=output_contract,
        aliases=tuple(aliases),
        required_kwargs=tuple(required_kwargs),
        default_kwargs=tuple((default_kwargs or {}).items()),
        offline_kwargs=offline_kwargs,
        notes=notes,
    )


_SPECS = (
    _spec(
        "unet",
        ".UNet",
        "UNet",
        "baseline",
        "concatenated",
        "damage_tensor",
        aliases=("UNet",),
        default_kwargs={"in_channels": 6, "num_classes": 4},
    ),
    _spec(
        "deeplab_v3_plus",
        ".DeepLabV3Plus",
        "DeepLabV3Plus",
        "baseline",
        "concatenated",
        "damage_tensor",
        aliases=("DeepLabV3Plus", "deeplabv3plus"),
        default_kwargs={"in_channels": 6, "num_classes": 4},
    ),
    _spec(
        "siam_attn_unet",
        ".SiamAttnUNet",
        "SiamAttnUNet",
        "baseline",
        aliases=("SiamAttnUNet",),
        default_kwargs={"in_channels": 3, "num_classes": 4},
        output_contract="damage_tensor",
    ),
    _spec(
        "siam_crnn",
        ".SiamCRNN",
        "SiamCRNN",
        "baseline",
        aliases=("SiamCRNN",),
    ),
    _spec(
        "damageformer",
        ".DamageFormer",
        "DamageFormer",
        "damageformer_baseline",
        aliases=("DamageFormer",),
        offline=True,
    ),
    _spec(
        "damageformer_bg",
        ".DamageFormer_BG",
        "DamageFormer_BG",
        "building_guidance",
        aliases=("DamageFormer_BG", "bg"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp",
        ".DamageFormer_BGMP",
        "DamageFormer_BGMP",
        "prototype_memory",
        aliases=("DamageFormer_BGMP", "bgmp"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp_autok",
        ".DamageFormer_BGMP_AutoK",
        "DamageFormer_BGMP_AutoK",
        "prototype_memory",
        aliases=("DamageFormer_BGMP_AutoK", "bgmp_autok"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp_aux",
        ".DamageFormer_BGMP_Aux",
        "DamageFormer_BGMP_Aux",
        "prototype_memory",
        aliases=("DamageFormer_BGMP_Aux", "bgmp_aux"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp_v2",
        ".DamageFormer_BGMP_V2",
        "DamageFormer_BGMP_V2",
        "prototype_memory",
        aliases=("DamageFormer_BGMP_V2", "bgmp_v2"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp_k3_gated",
        ".DamageFormer_BGMP_K3Gated",
        "DamageFormer_BGMP_K3Gated",
        "prototype_memory",
        aliases=("DamageFormer_BGMP_K3Gated", "bgmp_k3_gated"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp_k3_gated_ddp",
        ".DamageFormer_BGMP_K3Gated_DDP",
        "DamageFormer_BGMP_K3Gated_DDP",
        "prototype_memory",
        aliases=("DamageFormer_BGMP_K3Gated_DDP", "bgmp_k3_gated_ddp"),
        offline=True,
    ),
    _spec(
        "damageformer_bgmp_k3_event_balanced_ddp",
        ".DamageFormer_BGMP_K3EventBalanced_DDP",
        "DamageFormer_BGMP_K3EventBalanced_DDP",
        "prototype_memory",
        aliases=(
            "DamageFormer_BGMP_K3EventBalanced_DDP",
            "bgmp_k3_event_balanced_ddp",
        ),
        offline=True,
    ),
    _spec(
        "damageformer_s1b",
        ".DamageFormer_S1B",
        "DamageFormer_S1B",
        "staged_experiment",
        aliases=("DamageFormer_S1B", "s1b"),
        offline=True,
    ),
    _spec(
        "damageformer_s1c",
        ".DamageFormer_S1C",
        "DamageFormer_S1C",
        "staged_experiment",
        aliases=("DamageFormer_S1C", "s1c"),
        offline=True,
    ),
    _spec(
        "damageformer_s3_bgmp_k3",
        ".DamageFormer_S3_BGMP_K3",
        "DamageFormer_S3_BGMP_K3",
        "staged_experiment",
        aliases=("DamageFormer_S3_BGMP_K3", "s3_bgmp_k3"),
        offline=True,
    ),
    _spec(
        "damageformer_s5_hcdm",
        ".DamageFormer_S5_HCDM",
        "DamageFormer_S5_HCDM",
        "hierarchical_difference",
        aliases=("DamageFormer_S5_HCDM", "s5_hcdm"),
        offline=True,
    ),
    _spec(
        "damageformer_hierarchical_rgbsar",
        ".DamageFormer_Hierarchical_RGBSAR",
        "DamageFormer_Hierarchical_RGBSAR",
        "hierarchical_difference",
        "paired",
        "mapping_or_tuple",
        aliases=("DamageFormer_Hierarchical_RGBSAR", "hierarchical_rgbsar"),
        offline=True,
    ),
    _spec(
        "damageformer_s8_ia_mla",
        ".DamageFormer_S8_IA_MLA",
        "DamageFormer_S8_IA_MLA",
        "hierarchical_difference",
        "paired",
        "mapping_or_tuple",
        aliases=("DamageFormer_S8_IA_MLA", "s8_ia_mla"),
        offline=True,
    ),
    _spec(
        "damageformer_priority_modular",
        ".DamageFormer_Priority_Modular",
        "DamageFormerPriorityModular",
        "instance_reasoning",
        "paired",
        "mapping",
        aliases=("DamageFormerPriorityModular", "priority_modular"),
        offline=True,
    ),
    _spec(
        "damageformer_darm",
        ".DamageFormer_DARM",
        "DamageFormer_DARM",
        "instance_reasoning",
        "paired",
        "mapping",
        aliases=("DamageFormer_DARM", "darm"),
        offline=True,
    ),
    _spec(
        "damageformer_iarm",
        ".DamageFormer_IARM",
        "DamageFormer_IARM",
        "instance_reasoning",
        "paired",
        "custom_tuple",
        aliases=("DamageFormer_IARM", "iarm"),
        offline=True,
    ),
    _spec(
        "damageformer_iarm_e2a",
        ".DamageFormer_IARM_E2A",
        "DamageFormer_IARM_E2A",
        "instance_reasoning",
        "paired",
        "custom_tuple",
        aliases=("DamageFormer_IARM_E2A", "iarm_e2a"),
        offline=True,
    ),
    _spec(
        "damageformer_s7_sirm",
        ".DamageFormer_S7_SIRM",
        "DamageFormer_S7_SIRM",
        "retrieval_wrapper",
        "paired",
        "custom",
        aliases=("DamageFormer_S7_SIRM", "s7_sirm"),
        required_kwargs=("base_model", "memory", "config"),
        notes="Retrieval wrapper; construct its base model and memory explicitly.",
    ),
    _spec(
        "damageformer_s7_sirm_m2b",
        ".DamageFormer_S7_SIRM_M2B",
        "DamageFormer_S7_SIRM_M2B",
        "retrieval_wrapper",
        "paired",
        "custom",
        aliases=("DamageFormer_S7_SIRM_M2B", "s7_sirm_m2b"),
        required_kwargs=("base_model", "memory", "config"),
        notes="Retrieval wrapper; construct its base model and memory explicitly.",
    ),
    _spec(
        "damageformer_s8_micb_dsmil",
        ".DamageFormer_S8_MICB_DSMIL",
        "DamageFormer_S8_MICB_DSMIL",
        "instance_reasoning",
        "paired",
        "mapping_or_tuple",
        aliases=("DamageFormer_S8_MICB_DSMIL", "s8_micb_dsmil"),
    ),
    _spec(
        "damageformer_rgbsar_ccu_bda",
        ".DamageFormer_RGBSAR_CCU_BDA",
        "DamageFormerRGBSARCCUBDA",
        "heterogeneous_change_conditioning",
        "paired_rgb_native_sar",
        "mapping_or_tuple",
        aliases=("DamageFormerRGBSARCCUBDA", "ccu_bda", "ccu"),
        default_kwargs={"variant": "c3", "pretrained_rgb": False},
        notes=(
            "Independent RGB/native-SAR encoders with building-local "
            "correlation and change-conditioned unification at stages 3/4."
        ),
    ),
    _spec(
        "damageformer_rgbsar_h12_dgb",
        ".DamageFormer_RGBSAR_H12_DGB",
        "DamageFormerRGBSARH12DGB",
        "heterogeneous_change_conditioning",
        "paired_rgb_native_sar",
        "mapping_or_tuple",
        aliases=("DamageFormerRGBSARH12DGB", "h12_dgb", "h12"),
        default_kwargs={"variant": "d1", "pretrained_rgb": False},
        notes=(
            "Locked H6-A2 with a zero-initialised Stage2/3 damage-gate bypass. "
            "Edits Intact vs Affected only; localisation and severity odds stay frozen."
        ),
    ),
    _spec(
        "mean_teacher",
        ".MeanTeacher.network",
        "MTNetwork",
        "domain_adaptation",
        "concatenated",
        "custom",
        aliases=("MTNetwork", "MeanTeacher"),
        required_kwargs=("num_classes", "criterion", "norm_layer"),
        notes="Already contains its own EMA teacher/student branches.",
    ),
)


MODEL_SPECS: Dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}


def _normalize_name(name: str) -> str:
    return name.strip().replace("-", "_").replace(" ", "_").lower()


def _alias_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    for spec in _SPECS:
        for alias in (spec.name, spec.class_name, *spec.aliases):
            key = _normalize_name(alias)
            previous = index.get(key)
            if previous is not None and previous != spec.name:
                raise RuntimeError(
                    f"Duplicate model alias {alias!r}: {previous!r} and {spec.name!r}"
                )
            index[key] = spec.name
    return index


_ALIASES = _alias_index()


def resolve_model_name(name: str) -> str:
    """Return the canonical registry name for a name, class, or alias."""

    key = _normalize_name(name)
    try:
        return _ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model {name!r}. Available models: {choices}") from exc


def get_model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[resolve_model_name(name)]


def list_model_specs(*, family: str | None = None) -> Tuple[ModelSpec, ...]:
    """List specs in stable registry order, optionally filtered by family."""

    specs = _SPECS
    if family is not None:
        family_key = _normalize_name(family)
        specs = tuple(
            spec for spec in specs if _normalize_name(spec.family) == family_key
        )
    return tuple(specs)


def create_model(name: str, *, offline: bool = False, **kwargs: Any) -> Any:
    """Construct a registered model.

    ``offline=True`` disables pretrained Swin initialization for implementations
    that expose ``use_pretrained_swin``. Explicit keyword arguments always win.
    """

    spec = get_model_spec(name)
    constructor_kwargs = spec.constructor_defaults()
    if offline:
        for key, value in spec.offline_defaults().items():
            constructor_kwargs.setdefault(key, value)
    constructor_kwargs.update(kwargs)

    missing = [key for key in spec.required_kwargs if key not in constructor_kwargs]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"{spec.name!r} requires constructor arguments: {joined}. "
            "This model is a configured wrapper rather than a standalone baseline."
        )

    model_class = spec.load_class()
    return model_class(**constructor_kwargs)
