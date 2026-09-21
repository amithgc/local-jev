import numpy as np
import pytest

from local_jev import primitives as P


def test_confidence_matches_jevs_three_option_formula():
    # Jev's docs: (3 * p_max - 1) / 2. A real Jev response had p_max 0.79 -> confidence 0.68.
    assert P.confidence([0.79, 0.0, 0.21]) == pytest.approx((3 * 0.79 - 1) / 2)
    assert abs(P.confidence([0.79, 0.0, 0.21]) - 0.68) < 0.01      # 0.79 was itself rounded by the API


def test_confidence_bounds():
    assert P.confidence([1.0, 0.0, 0.0, 0.0]) == 1.0
    assert P.confidence([0.25] * 4) == 0.0
    assert P.confidence([1.0]) == 1.0


def test_score_is_the_expected_level():
    # The worked example in Jev's docs: 0.57 / 0.43 on levels 1 and 2 -> 1.43.
    answer = P.score_answer(["low", "mid", "high"], [0.0, 0.57, 0.43])
    assert answer["score"] == pytest.approx(1.43)
    assert answer["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert set(answer["probabilities"]) == {"0", "1", "2"}


def test_choice_answer_shape():
    answer = P.choice_answer(["billing", "technical", "sales"], [0.21, 0.79, 0.0])
    assert answer == {"type": "choice", "choice": "technical", "confidence": 0.685,
                      "probabilities": {"billing": 0.21, "technical": 0.79, "sales": 0.0}}


def test_noul_has_no_confidence_field():
    assert P.noul_answer(0.9312345) == {"type": "noul", "noul": 0.9312}


def test_softmax_temperature_flattens():
    sharp, flat = P.softmax([2.0, 0.0], 1.0), P.softmax([2.0, 0.0], 4.0)
    assert sharp[0] > flat[0] > 0.5 and np.isclose(flat.sum(), 1.0)


def test_calibration_flattens_but_never_reorders():
    from local_jev.engine import calibrate
    assert np.allclose(calibrate([0.8, 0.2]), [0.8, 0.2])
    cooled = calibrate([0.7, 0.2, 0.1], temperature=3.0)
    assert list(np.argsort(cooled)) == [2, 1, 0] and 1 / 3 < cooled[0] < 0.7 and np.isclose(cooled.sum(), 1.0)
    logit = np.log(0.8 / 0.2)
    assert np.isclose(calibrate([0.8, 0.2], 2.0)[0], 1 / (1 + np.exp(-logit / 2)))      # for a yes/no it is sigmoid(logit / T)
