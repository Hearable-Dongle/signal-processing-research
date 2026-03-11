from pathlib import Path

from localization.tuning.common import (
    SceneGeometry,
    balanced_robustness_score,
    build_eval_config,
    stratified_scene_subset,
    window_ms_to_nfft,
)


def test_window_ms_to_nfft_uses_sample_rate():
    assert window_ms_to_nfft(10, 16000) == 160
    assert window_ms_to_nfft(250, 16000) == 4000


def test_build_eval_config_maps_window_to_nfft():
    cfg = build_eval_config(
        base_cfg={"nfft": 512, "overlap": 0.5, "freq_range": [200, 3000]},
        fs=16000,
        window_ms=25,
        overlap=0.75,
        freq_low_hz=300,
        freq_high_hz=4200,
        extra_updates={"epsilon": 0.15},
    )
    assert cfg["window_ms"] == 25
    assert cfg["nfft"] == 400
    assert cfg["overlap"] == 0.75
    assert cfg["freq_range"] == [300, 4200]
    assert cfg["epsilon"] == 0.15


def test_stratified_scene_subset_samples_each_bucket():
    scenes = []
    idx = 0
    for angle in (0, 45):
        for layout in ("single", "opposite_pair"):
            for dup in range(3):
                scenes.append(
                    SceneGeometry(
                        scene_id=f"scene_{idx}",
                        scene_path=Path(f"/tmp/scene_{idx}.json"),
                        metadata_path=None,
                        scene_type="testing_specific_angles",
                        main_angle_deg=angle,
                        secondary_angle_deg=(angle + 45) % 360,
                        noise_layout_type=layout,
                        noise_angles_deg=(dup * 45,),
                    )
                )
                idx += 1

    selected = stratified_scene_subset(scenes, per_bucket=1, seed=123)
    assert len(selected) == 4
    assert {(scene.main_angle_deg, scene.noise_layout_type) for scene in selected} == {
        (0, "single"),
        (0, "opposite_pair"),
        (45, "single"),
        (45, "opposite_pair"),
    }


def test_balanced_robustness_score_prefers_stable_accurate_results():
    strong = balanced_robustness_score(
        mae_deg=4.0,
        acc10=0.9,
        recall=0.95,
        misses_mean=0.1,
        false_alarms_mean=0.0,
        angle_std=1.5,
        layout_std=2.0,
    )
    weak = balanced_robustness_score(
        mae_deg=18.0,
        acc10=0.45,
        recall=0.6,
        misses_mean=1.2,
        false_alarms_mean=0.8,
        angle_std=7.0,
        layout_std=9.0,
    )
    assert strong > weak
