"""Binary math reward shared by SVD analysis and GRPO training.

There is no separate format reward. The last box in the complete response is
verified; if there is no box, math_verify extracts an answer from the complete
response.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from verl.utils.reward_score import math_dapo


REWARD_VERIFIER = "math_verify_last_boxed_or_full_fallback_v1"
BOXED_SEARCH_SCOPE = "full_response"
_MATH_VERIFIER = None


def _build_math_verifier():
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    for logger_name in ("math_verify.metric", "math_verify.grader", "math_verify.parser"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def _get_math_verifier():
    global _MATH_VERIFIER
    if _MATH_VERIFIER is None:
        _MATH_VERIFIER = _build_math_verifier()
    return _MATH_VERIFIER


def _json_safe_prediction(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonicalize_ground_truth(ground_truth: str) -> str:
    return re.sub(r"(\\boxed)\s*\{", r"\1{", ground_truth)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    del data_source, extra_info

    ground_truth = _canonicalize_ground_truth(str(ground_truth))
    if not ground_truth.strip():
        return {
            "score": -1.0,
            "acc": False,
            "pred": "[INVALID]",
            "reward_verifier": REWARD_VERIFIER,
            "verification_status": "missing_ground_truth",
            "invalid": True,
            "verification_error": False,
        }

    solution_str = str(solution_str)
    boxed_prediction = math_dapo.last_boxed_only_string(solution_str)
    normalized_ground_truth = math_dapo.normalize_final_answer(ground_truth)
    prediction: Any = "[INVALID]"

    if boxed_prediction is not None:
        extracted_prediction = math_dapo.remove_boxed(boxed_prediction)
        prediction = math_dapo.normalize_final_answer(extracted_prediction)
        if prediction == normalized_ground_truth:
            return {
                "score": 1.0,
                "acc": True,
                "pred": prediction,
                "reward_verifier": REWARD_VERIFIER,
                "verification_status": "ok",
                "invalid": False,
                "verification_error": False,
            }
        candidate_output = boxed_prediction
    else:
        candidate_output = solution_str

    stripped_ground_truth = ground_truth.strip()
    boxed_ground_truth = math_dapo.last_boxed_only_string(stripped_ground_truth)
    if boxed_ground_truth != stripped_ground_truth:
        boxed_ground_truth = f"\\boxed{{{stripped_ground_truth}}}"

    try:
        score, extracted = _get_math_verifier()([boxed_ground_truth], [candidate_output])
        correct = bool(score)
        extracted_predictions = []
        if extracted is not None and len(extracted) > 1 and extracted[1]:
            extracted_predictions = extracted[1]
            if boxed_prediction is None:
                prediction = extracted_predictions[-1]
        invalid = not bool(extracted_predictions)
        verification_error = False
        verification_status = "unparseable_prediction" if invalid else "ok"
    except Exception as exc:
        correct = False
        invalid = True
        verification_error = True
        verification_status = f"error:{type(exc).__name__}"

    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": _json_safe_prediction(prediction),
        "reward_verifier": REWARD_VERIFIER,
        "verification_status": verification_status,
        "invalid": invalid,
        "verification_error": verification_error,
    }


def _self_test() -> None:
    long_trailing_response = (
        r"Therefore, the answer is \boxed{72}."
        + " This trailing verification text is intentionally irrelevant." * 12
    )
    if r"\boxed" in long_trailing_response[-300:]:
        raise SystemExit(
            "Long-response self-test does not place boxed outside the final 300 chars"
        )
    cases = (
        (r"Therefore, the answer is \boxed{72}.", "72", True, False),
        ("Therefore, the answer is 72.", "72", True, False),
        (long_trailing_response, "72", True, False),
        (r"The answer is \boxed{-\frac{8}{12}}.", r"-\frac{2}{3}", True, False),
        (r"The answer is \boxed{71}.", "72", False, False),
        ("No final answer was produced.", "72", False, True),
    )
    for response, ground_truth, expected_accuracy, expected_invalid in cases:
        result = compute_score("math", response, ground_truth)
        if (
            bool(result["acc"]) != expected_accuracy
            or bool(result["invalid"]) != expected_invalid
        ):
            raise SystemExit(
                f"Reward self-test failed: response={response!r}, "
                f"ground_truth={ground_truth!r}, result={result!r}"
            )
    print(f"Reward self-test passed: {len(cases)} cases ({REWARD_VERIFIER})")


if __name__ == "__main__":
    _self_test()
