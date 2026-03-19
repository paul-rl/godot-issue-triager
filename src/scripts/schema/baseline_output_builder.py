class BaselineOutputBuilder:
    """
    Builds schema-compliant triage outputs from baseline scores.
    Routing decision (needs_human_triage) is based on COMPONENTS.
    """

    def __init__(
        self,
        *,
        schema_version: str = "1.0",
        thresholds: dict[str, float] | None = None,
        tau_components: float = 0.5,
        margin: float = 0.0
    ):
        self.schema_version = schema_version
        self.thresholds = thresholds or {}
        self.tau_components = tau_components
        self.margin = margin

    @staticmethod
    def _pick(scores: dict[str, float], t: float, max_items: int | None = None) -> list[str]:
        chosen = [
            k for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if v >= t
        ]
        if max_items is not None:
            chosen = chosen[:max_items]
        return chosen

    def _needs_human_triage(self, component_scores, pick_threshold):
        sorted_components = sorted(component_scores.items(), key=lambda kv: kv[1], reverse=True)
        top1 = sorted_components[0][1] if sorted_components else 0.0
        top2 = sorted_components[1][1] if len(sorted_components) > 1 else 0.0
        return (top1 < pick_threshold) or ((top1 - top2) < self.margin)

    def build(
        self,
        *,
        issue_type_scores: dict[str, float],
        component_scores: dict[str, float],
        platform_scores: dict[str, float],
        impact_scores: dict[str, float],
        meta: dict | None = None,
    ) -> dict:
        th = self.thresholds

        labels = {
            "issue_type": self._pick(issue_type_scores, th.get("issue_type", 0.5), max_items=1),
            "components": self._pick(component_scores, th.get("components", self.tau_components)),
            "platform": self._pick(platform_scores, th.get("platform", 0.5)),
            "impact": self._pick(impact_scores, th.get("impact", 0.5)),
        }

        pick_threshold = th.get("components", self.tau_components)
        needs_human = self._needs_human_triage(component_scores, pick_threshold) or (len(labels["components"]) == 0)
        out = {
            "schema_version": self.schema_version,
            "labels": labels,
            "needs_human_triage": needs_human,
            "scores": {
                "issue_type": issue_type_scores,
                "components": component_scores,
                "platform": platform_scores,
                "impact": impact_scores,
            },
            "meta": meta or {"source": "baseline"},
        }

        return out
