An NLP Pipeline which ingests issues from the Godot engine repo and converts them into structured routing records. This enables faster, more consistent assignment and prioritization with measurable reliability.

The schema will look as follows:
```json
{
    "issue_type": "bug" | "feature_request" | "discussion" | "docs" | "other",
    "components": ["rendering", "gui", ...],
    "platform": ["windows|linuxbsd|macos|android|web|unknow"],
    "impact": ["high_priority", "regression", ...],
    "needs_human_traige": false,
    "confidence": 0.0
}```