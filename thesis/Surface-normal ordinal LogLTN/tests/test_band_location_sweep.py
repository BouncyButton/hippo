from argparse import Namespace

import run_band_location_sweep as sweep


def _metric(mean: float) -> dict[str, float]:
    return {"mean": mean}


def _comparison(*, surface: float, union: float, assd: float, errors: float) -> dict:
    return {
        "surface_dice_1mm": _metric(surface),
        "union_dice": _metric(union),
        "assd_mm": _metric(assd),
        "total_errors": _metric(errors),
    }


def test_smallest_ratio_passing_two_updates_is_selected() -> None:
    args = Namespace(
        location_shares=[0.05, 0.10],
        max_union_dice_drop=0.0005,
        minimum_passing_update_count=2,
    )
    summary = {
        "0.02": {
            "comparisons_vs_band": {
                "rho_05": _comparison(surface=0.001, union=0, assd=-0.001, errors=-1),
                "rho_10": _comparison(surface=0.002, union=0, assd=-0.002, errors=-2),
            }
        },
        "0.05": {
            "comparisons_vs_band": {
                "rho_05": _comparison(surface=0.001, union=0, assd=-0.001, errors=0),
                "rho_10": _comparison(surface=0.002, union=0, assd=-0.002, errors=-2),
            }
        },
        "0.1": {
            "comparisons_vs_band": {
                "rho_05": _comparison(surface=-0.001, union=0, assd=0, errors=0),
                "rho_10": _comparison(surface=0.003, union=0, assd=-0.003, errors=-3),
            }
        },
    }
    decision = sweep._decision(summary, args)
    assert decision["verdict"] == "GO_TO_FIVE_EPOCH_PILOT"
    assert decision["selected_location_share"] == 0.05


def test_ratio_fails_when_surface_gain_costs_errors() -> None:
    args = Namespace(
        location_shares=[0.05],
        max_union_dice_drop=0.0005,
        minimum_passing_update_count=2,
    )
    comparison = _comparison(surface=0.001, union=0, assd=-0.001, errors=1)
    summary = {
        value: {"comparisons_vs_band": {"rho_05": comparison}}
        for value in ("0.02", "0.05", "0.1")
    }
    decision = sweep._decision(summary, args)
    assert decision["verdict"] == "NO_GO"
    assert decision["selected_location_share"] is None
