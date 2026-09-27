"""Check Run content. Conclusion is always `neutral` (or `skipped` on fail-open) so Preflight
can never block a merge (SPEC: fail-open, never blocks)."""

from typing import Any

CHECK_NAME = "Preflight"
ICON = {"low": "🟢", "watch": "🟡", "high": "🔴"}
LABEL = {"low": "Low risk", "watch": "Watch", "high": "High risk"}
ACTIONS = [
    {"label": "Useful", "description": "This check helped", "identifier": "useful"},
    {"label": "Not useful", "description": "This check was noise", "identifier": "not_useful"},
]


def title(score: dict[str, Any]) -> str:
    if score.get("abstain"):
        return f"⚪ Not enough signal · {score['p_fail']:.0%} estimated failure risk"
    return f"{ICON[score['band']]} {LABEL[score['band']]} · {score['p_fail']:.0%} chance CI fails"


def summary(score: dict[str, Any]) -> str:
    lines = []
    if score.get("abstain"):
        lines.append(
            "The model's conformal prediction set covers both outcomes, so it won't "
            "guess. CI runs as usual."
        )
    if score.get("reasons"):
        lines += ["", "**Why**"]
        lines += [
            f"- {'↑' if r['direction'] == 'raises' else '↓'} {r['text']}" for r in score["reasons"]
        ]
    if score.get("what_ifs"):
        lines += ["", "**What would lower the risk**", "", "| Change | Risk |", "|---|---|"]
        lines += [
            f"| {w['text']} | {w['p_before']:.0%} → **{w['p_after']:.0%}** |"
            for w in score["what_ifs"]
        ]
    lines += ["", f"**CI plan:** `{score['ci_plan']}`"]
    if score.get("test_order"):
        top = score["test_order"][:5]
        lines += ["", "**Tests most likely to fail (run first)**"]
        lines += [f"{i}. `{t}`" for i, t in enumerate(top, 1)]
    lines += [
        "",
        f"<sub>model {score['model_version']} ({score['model_key']}) · "
        f"{score['latency_ms']:.0f} ms · Preflight never blocks a merge</sub>",
    ]
    return "\n".join(lines)


def check_run_payload(head_sha: str, score: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "neutral",
        "output": {"title": title(score), "summary": summary(score)},
        "actions": ACTIONS,
    }


def skipped_payload(head_sha: str, reason: str) -> dict[str, Any]:
    return {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "skipped",
        "output": {
            "title": "Preflight skipped · CI runs normally",
            "summary": f"Risk scoring was skipped ({reason}). Nothing is blocked; the full CI "
            "pipeline runs as usual.",
        },
    }
