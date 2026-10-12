from pathlib import Path

from bda_benchmark.model.registry import (
    get_model_spec,
    list_model_specs,
    resolve_model_name,
)


def test_registry_names_and_aliases_are_unique():
    specs = list_model_specs()
    assert len(specs) == len({spec.name for spec in specs})
    for spec in specs:
        assert resolve_model_name(spec.name) == spec.name
        assert resolve_model_name(spec.class_name) == spec.name


def test_registered_modules_exist():
    model_dir = Path(__file__).resolve().parents[1]
    for spec in list_model_specs():
        relative_module = spec.module.removeprefix(".").replace(".", "/")
        assert (model_dir / f"{relative_module}.py").is_file()


def test_short_alias_resolves():
    assert get_model_spec("s1c").name == "damageformer_s1c"
    assert get_model_spec("UNet").input_mode == "concatenated"

