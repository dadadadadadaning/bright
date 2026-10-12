import numpy as np

from s9_watershed_common import (
    WatershedConfig,
    labels_to_instances_and_specs,
    probability_seed_watershed_labels,
)


def make_two_touching_buildings():
    yy, xx = np.mgrid[:96, :96]
    left = (yy - 48) ** 2 + (xx - 34) ** 2 <= 18 ** 2
    right = (yy - 48) ** 2 + (xx - 62) ** 2 <= 18 ** 2

    probability = np.zeros((96, 96), dtype=np.float32)
    probability[left] = np.maximum(
        probability[left],
        0.92,
    )
    probability[right] = np.maximum(
        probability[right],
        0.90,
    )

    # Low-confidence semantic bridge: region keeps it, seeds do not.
    probability[45:52, 47:50] = 0.55
    return probability


def main():
    probability = make_two_touching_buildings()
    config = WatershedConfig(
        name="synthetic",
        region_threshold=0.50,
        seed_threshold=0.80,
        min_peak_distance=10,
        min_instance_area=32,
    )
    labels, diagnostics = probability_seed_watershed_labels(
        probability,
        config,
    )
    instances, specs = labels_to_instances_and_specs(
        labels,
        batch_index=0,
    )

    assert len(instances) == len(specs)
    assert len(instances) >= 2, (
        diagnostics,
        len(instances),
    )
    assert all(instance.area >= 32 for instance in instances)
    print(
        "Synthetic S9-A2 watershed test: PASS",
        diagnostics,
    )


if __name__ == "__main__":
    main()
