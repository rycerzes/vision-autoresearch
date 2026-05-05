"""Promotion policy: load from YAML config and compare candidate vs baseline.

``tie_breakers`` compare candidate vs baseline with **higher-is-better** on
each listed metric (use for auxiliary metrics even when ``primary`` is
lower-is-better).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vision_lab.metrics import MetricDirection, assert_standard_metric_name, direction_for_standard_metric
from vision_lab.summary_schema import validate_summary_keys_for_task
from vision_lab.task_registry import TASK_BY_ID, get_task, promotion_metric_for_task


@dataclass(frozen=True)
class GateSpec:
    """Absolute threshold on a candidate metric value."""

    metric: str
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class PromotionPolicy:
    """Resolved promotion settings for one benchmark run."""

    primary: str
    direction: MetricDirection
    min_delta: float
    secondary: str | None
    gates: tuple[GateSpec, ...]
    tie_breakers: tuple[str, ...]


@dataclass(frozen=True)
class PromotionEvaluation:
    """Outcome of comparing a candidate run against the current master."""

    promoted: bool
    primary: str
    direction: MetricDirection
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    relative_delta: float | None
    min_delta: float
    min_delta_met: bool
    gates_met: bool
    rerun_recommended: bool
    reason: str


_REL_EPS = 1e-12


def _parse_float(val: Any) -> float | None:
    if val in (None, ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _default_direction_for_metric(metric: str) -> MetricDirection:
    return direction_for_standard_metric(metric)


def _assert_metric_allowed_for_task(task_id: str, metric: str, role: str) -> None:
    assert_standard_metric_name(metric)
    spec = get_task(task_id)
    role_allowed: frozenset[str]
    if role == "primary":
        role_allowed = spec.allowed_primary_metrics
    elif role == "secondary":
        role_allowed = spec.allowed_secondary_metrics
    elif role == "gates":
        role_allowed = spec.allowed_gate_metrics
    elif role == "tie_breakers":
        role_allowed = spec.allowed_tie_breaker_metrics
    else:
        raise AssertionError(f"unknown promotion role: {role!r}")
    if metric not in role_allowed:
        raise ValueError(
            f"promotion {role} metric {metric!r} is not allowed for task {task_id!r} "
            f"(allowed for this role: {', '.join(sorted(role_allowed))})."
        )


def validate_promotion_policy_for_task(task_id: str, policy: PromotionPolicy) -> None:
    """Ensure every promotion field references only task-permitted standard metrics."""
    _assert_metric_allowed_for_task(task_id, policy.primary, "primary")
    if policy.secondary:
        _assert_metric_allowed_for_task(task_id, policy.secondary, "secondary")
    for g in policy.gates:
        _assert_metric_allowed_for_task(task_id, g.metric, "gates")
    for tb in policy.tie_breakers:
        _assert_metric_allowed_for_task(task_id, tb, "tie_breakers")


def promotion_dependency_metric_names(policy: PromotionPolicy) -> tuple[str, ...]:
    """Metrics referenced by the resolved promotion policy (deduped, stable order)."""
    ordered: list[str] = [policy.primary]
    if policy.secondary:
        ordered.append(policy.secondary)
    ordered.extend(g.metric for g in policy.gates)
    ordered.extend(policy.tie_breakers)
    seen: set[str] = set()
    out: list[str] = []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _require_numeric_summary_metric(
    summary_metrics: Mapping[str, Any],
    metric: str,
    *,
    task_id: str,
    role: str,
) -> None:
    raw = summary_metrics.get(metric)
    if raw is None:
        raise ValueError(
            f"Summary is missing {role} metric {metric!r} for task {task_id!r}; "
            "runs without every promotion dependency metric are not recorded."
        )
    if isinstance(raw, bool):
        raise ValueError(
            f"{role.capitalize()} metric {metric!r} must be numeric in the summary; got {raw!r}."
        )
    if isinstance(raw, (int, float)):
        return
    if isinstance(raw, str):
        try:
            float(raw)
        except ValueError:
            raise ValueError(
                f"{role.capitalize()} metric {metric!r} must be numeric in the summary; got {raw!r}."
            ) from None
        return
    raise ValueError(
        f"{role.capitalize()} metric {metric!r} must be numeric in the summary; got {raw!r}."
    )


def assert_summary_eligible_for_recording(
    *,
    task_id: str,
    policy: PromotionPolicy,
    summary_metrics: Mapping[str, Any],
) -> None:
    """Require every promotion dependency metric to be present and numeric (strict recording gate)."""
    validate_summary_keys_for_task(task_id, summary_metrics)
    validate_promotion_policy_for_task(task_id, policy)

    secondary_set = {policy.secondary} if policy.secondary else set()
    gate_set = {g.metric for g in policy.gates}
    tb_set = set(policy.tie_breakers)

    for name in promotion_dependency_metric_names(policy):
        if name == policy.primary:
            role = "promotion primary"
        elif name in secondary_set:
            role = "promotion secondary"
        elif name in gate_set:
            role = "promotion gate"
        elif name in tb_set:
            role = "promotion tie_breaker"
        else:
            role = "promotion"
        _require_numeric_summary_metric(
            summary_metrics, name, task_id=task_id, role=role
        )


def _task_default_policy(task_id: str) -> PromotionPolicy:
    primary = promotion_metric_for_task(task_id)
    policy = PromotionPolicy(
        primary=primary,
        direction=_default_direction_for_metric(primary),
        min_delta=0.0,
        secondary=None,
        gates=(),
        tie_breakers=(),
    )
    validate_promotion_policy_for_task(task_id, policy)
    return policy


def load_promotion_policy(config_data: Mapping[str, Any], *, task_id: str) -> PromotionPolicy:
    """Build policy from optional ``promotion:`` block; otherwise task registry defaults."""
    if task_id not in TASK_BY_ID:
        raise ValueError(f"Unknown task: {task_id!r}")
    if "promotion_metric" in config_data:
        raise ValueError(
            "Unsupported config key 'promotion_metric'. Use a top-level `promotion:` mapping "
            "(see AGENTS.md), or omit it to use task defaults from vision_lab.task_registry."
        )

    block = config_data.get("promotion")
    if block is None:
        return _task_default_policy(task_id)
    if not isinstance(block, dict):
        raise ValueError("promotion must be a mapping when set")
    if not block:
        return _task_default_policy(task_id)

    primary = block.get("primary")
    if primary is None or str(primary).strip() == "":
        primary = promotion_metric_for_task(task_id)
    primary = str(primary).strip()

    dir_raw = block.get("direction")
    if dir_raw is None or str(dir_raw).strip() == "":
        direction = _default_direction_for_metric(primary)
    else:
        direction = MetricDirection(str(dir_raw).strip().lower())

    min_delta_raw = block.get("min_delta", 0.0)
    try:
        min_delta = float(min_delta_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"promotion.min_delta must be numeric; got {min_delta_raw!r}") from exc

    sec = block.get("secondary")
    secondary = str(sec).strip() if sec not in (None, "") else None
    spec = get_task(task_id)
    if secondary and not spec.allowed_secondary_metrics:
        raise ValueError(
            f"promotion.secondary is set but task {task_id!r} does not allow a secondary metric."
        )

    gates_raw = block.get("gates") or []
    if not isinstance(gates_raw, list):
        raise ValueError("promotion.gates must be a list of mappings")
    gates: list[GateSpec] = []
    for i, g in enumerate(gates_raw):
        if not isinstance(g, dict):
            raise ValueError(f"promotion.gates[{i}] must be a mapping")
        m = g.get("metric")
        if not m:
            raise ValueError(f"promotion.gates[{i}] missing metric")
        gmin = g.get("min")
        gmax = g.get("max")
        gates.append(
            GateSpec(
                metric=str(m).strip(),
                min=float(gmin) if gmin not in (None, "") else None,
                max=float(gmax) if gmax not in (None, "") else None,
            )
        )

    tb_raw = block.get("tie_breakers") or []
    if isinstance(tb_raw, str):
        tb_raw = [tb_raw]
    if not isinstance(tb_raw, list):
        raise ValueError("promotion.tie_breakers must be a list of metric names")
    tie_breakers = tuple(str(x).strip() for x in tb_raw if str(x).strip())

    if tie_breakers and not spec.allowed_tie_breaker_metrics:
        raise ValueError(
            f"promotion.tie_breakers is set but task {task_id!r} does not allow tie-breaker metrics."
        )

    policy = PromotionPolicy(
        primary=primary,
        direction=direction,
        min_delta=min_delta,
        secondary=secondary,
        gates=tuple(gates),
        tie_breakers=tie_breakers,
    )
    validate_promotion_policy_for_task(task_id, policy)
    return policy


def _marginal_improvement(
    *,
    candidate: float,
    baseline: float,
    direction: MetricDirection,
    min_delta: float,
) -> bool:
    """True when the run moved in the right direction but cleared less than min_delta."""
    if min_delta <= _REL_EPS:
        return False
    if direction == MetricDirection.HIGHER:
        return _REL_EPS < (candidate - baseline) < min_delta
    return _REL_EPS < (baseline - candidate) < min_delta


def _primary_beats(
    *,
    candidate: float,
    baseline: float,
    direction: MetricDirection,
    min_delta: float,
) -> bool:
    """Strict improvement when ``min_delta`` is 0 (ties do not promote)."""
    if direction == MetricDirection.HIGHER:
        gain = candidate - baseline
        if min_delta <= _REL_EPS:
            return gain > _REL_EPS
        return gain >= min_delta - _REL_EPS
    gain = baseline - candidate
    if min_delta <= _REL_EPS:
        return gain > _REL_EPS
    return gain >= min_delta - _REL_EPS


def _tie_break_winner(
    *,
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    tie_breakers: tuple[str, ...],
) -> bool | None:
    """Return True if candidate wins tie-break, False if baseline wins, None if still tied."""
    for tb in tie_breakers:
        c = _parse_float(candidate_metrics.get(tb))
        b = _parse_float(baseline_metrics.get(tb))
        if c is None or b is None:
            continue
        if abs(c - b) <= _REL_EPS:
            continue
        return c > b
    return None


def _baseline_metric_value(primary: str, baseline_row: Mapping[str, Any]) -> float | None:
    if baseline_row.get("promotion_metric") == primary:
        v = _parse_float(baseline_row.get("promotion_metric_value"))
        if v is not None:
            return v
    return _parse_float(baseline_row.get(primary))


def evaluate_promotion(
    *,
    policy: PromotionPolicy,
    candidate_metrics: Mapping[str, Any],
    baseline_row: Mapping[str, Any] | None,
) -> PromotionEvaluation:
    """Compare parsed summary metrics against the promoted master row."""
    primary = policy.primary
    candidate_value = _parse_float(candidate_metrics.get(primary))

    baseline_value: float | None
    if baseline_row is None:
        baseline_value = None
    else:
        baseline_value = _baseline_metric_value(primary, baseline_row)

    delta: float | None
    relative_delta: float | None
    if candidate_value is None:
        delta = None
        relative_delta = None
    elif baseline_value is None:
        delta = None
        relative_delta = None
    else:
        delta = candidate_value - baseline_value
        denom = max(abs(baseline_value), _REL_EPS)
        relative_delta = delta / denom

    gates_met = True
    for gate in policy.gates:
        v = _parse_float(candidate_metrics.get(gate.metric))
        if v is None:
            gates_met = False
            break
        if gate.min is not None and v < gate.min - _REL_EPS:
            gates_met = False
            break
        if gate.max is not None and v > gate.max + _REL_EPS:
            gates_met = False
            break

    min_delta_met = True
    promoted = False
    reason = ""

    if candidate_value is None:
        min_delta_met = False
        promoted = False
        reason = f"missing primary metric {primary!r}"
    elif baseline_value is None:
        min_delta_met = True
        promoted = bool(gates_met)
        reason = (
            f"first baseline for {primary}={candidate_value} (gates={'ok' if gates_met else 'failed'})"
            if promoted
            else f"gates failed for first baseline candidate {primary}={candidate_value}"
        )
    else:
        tie_tb: bool | None = None
        if abs(candidate_value - baseline_value) <= _REL_EPS and policy.tie_breakers:
            tie_tb = _tie_break_winner(
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_row or {},
                tie_breakers=policy.tie_breakers,
            )
            beats = tie_tb is True
            min_delta_met = beats
        else:
            beats = _primary_beats(
                candidate=candidate_value,
                baseline=baseline_value,
                direction=policy.direction,
                min_delta=policy.min_delta,
            )
            min_delta_met = beats

        promoted = bool(beats and gates_met)
        if not promoted:
            if not gates_met:
                reason = f"gates failed ({primary}: candidate={candidate_value}, baseline={baseline_value})"
            elif not beats:
                if tie_tb is False:
                    reason = "primary tied; baseline wins tie-break metrics"
                elif tie_tb is None and policy.tie_breakers and abs(candidate_value - baseline_value) <= _REL_EPS:
                    reason = "primary tied; tie-break inconclusive (missing or equal metrics)"
                elif policy.direction == MetricDirection.HIGHER:
                    reason = (
                        f"not enough gain: {primary} {candidate_value} vs baseline {baseline_value} "
                        f"(need +{policy.min_delta})"
                    )
                else:
                    reason = (
                        f"not enough gain: {primary} {candidate_value} vs baseline {baseline_value} "
                        f"(need drop {policy.min_delta})"
                    )
        else:
            reason = f"{primary} improved ({candidate_value} vs {baseline_value})"

    rerun_recommended = False
    if (
        not promoted
        and candidate_value is not None
        and baseline_value is not None
        and gates_met
    ):
        rerun_recommended = _marginal_improvement(
            candidate=candidate_value,
            baseline=baseline_value,
            direction=policy.direction,
            min_delta=policy.min_delta,
        )

    return PromotionEvaluation(
        promoted=promoted,
        primary=primary,
        direction=policy.direction,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        relative_delta=relative_delta,
        min_delta=policy.min_delta,
        min_delta_met=min_delta_met,
        gates_met=gates_met,
        rerun_recommended=rerun_recommended,
        reason=reason,
    )


def policy_to_jsonable(policy: PromotionPolicy) -> dict[str, Any]:
    return {
        "primary": policy.primary,
        "direction": policy.direction.value,
        "min_delta": policy.min_delta,
        "secondary": policy.secondary,
        "gates": [{"metric": g.metric, "min": g.min, "max": g.max} for g in policy.gates],
        "tie_breakers": list(policy.tie_breakers),
    }


def evaluation_to_jsonable(ev: PromotionEvaluation) -> dict[str, Any]:
    return {
        "promoted": ev.promoted,
        "primary": ev.primary,
        "direction": ev.direction.value,
        "baseline_value": ev.baseline_value,
        "candidate_value": ev.candidate_value,
        "delta": ev.delta,
        "relative_delta": ev.relative_delta,
        "min_delta": ev.min_delta,
        "min_delta_met": ev.min_delta_met,
        "gates_met": ev.gates_met,
        "rerun_recommended": ev.rerun_recommended,
        "reason": ev.reason,
    }
