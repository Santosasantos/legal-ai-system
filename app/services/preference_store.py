"""
Preference Store — Operator Edit Learning
──────────────────────────────────────────
The improvement loop:
  1. Operator submits (original_draft, edited_draft)
  2. We diff them to find what changed
  3. LLM analyses the diff and extracts reusable preference rules
  4. Rules are stored and injected into future draft generation prompts
  5. Rules accumulate and are deduplicated over time

This is a real improvement loop — not a version diff display.
"""

import json
import re
import difflib
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    OperatorEdit, PreferenceRule, PreferenceRuleStore,
)
from app.services.llm_provider import get_provider

logger = get_logger(__name__)


# ── Persistence ───────────────────────────────────────────────────────────────

def _rules_path() -> Path:
    p = Path(settings.PREFERENCE_RULES_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_preference_rules() -> List[PreferenceRule]:
    """Load all stored preference rules."""
    path = _rules_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        store = PreferenceRuleStore(**data)
        return store.rules
    except Exception as e:
        logger.warning(f"Could not load preference rules: {e}")
        return []


def save_preference_rules(rules: List[PreferenceRule]) -> None:
    """Persist preference rules to disk."""
    store = PreferenceRuleStore(rules=rules, last_updated=datetime.utcnow())
    _rules_path().write_text(store.model_dump_json(indent=2))
    logger.info(f"Saved {len(rules)} preference rules")


# ── Diff computation ──────────────────────────────────────────────────────────

def _compute_diff(original: str, edited: str) -> str:
    """
    Compute a human-readable unified diff between original and edited draft.
    Returns only the changed lines with context.
    """
    orig_lines = original.splitlines(keepends=True)
    edit_lines = edited.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, edit_lines,
        fromfile="original_draft",
        tofile="edited_draft",
        n=2,  # 2 lines of context
    ))
    return "".join(diff[:200])  # cap at 200 lines to stay in context


def _extract_additions_deletions(diff: str) -> Tuple[List[str], List[str]]:
    """Extract added and removed lines from a unified diff."""
    additions = []
    deletions = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            deletions.append(line[1:].strip())
    return additions, deletions


# ── Rule extraction via LLM ───────────────────────────────────────────────────

RULE_EXTRACTION_SYSTEM = """You are an expert at analysing how legal professionals edit AI-generated drafts.
Your job is to extract reusable writing preferences and style rules from the differences between
an original AI draft and an operator's edited version.

Focus on PATTERNS, not specific facts. Extract rules that would improve ALL future drafts.
Return ONLY valid JSON. No markdown, no explanation."""

RULE_EXTRACTION_PROMPT = """An operator edited an AI-generated legal draft. Analyse the changes and extract reusable preference rules.

ORIGINAL DRAFT (excerpt):
{original_excerpt}

EDITED DRAFT (excerpt):
{edited_excerpt}

DIFF (what changed):
{diff}

ADDITIONS (what the operator added):
{additions}

DELETIONS (what the operator removed):
{deletions}

Extract 2-5 reusable preference rules from these edits. Each rule should be a general instruction
that would improve future drafts — not specific to this document.

Return JSON array:
[
  {{
    "rule_text": "Always include the case number in the document header",
    "rule_category": "structure",
    "confidence": 0.9
  }},
  ...
]

Categories: style | structure | content | tone | format
Only include rules with confidence >= 0.6."""


async def extract_rules_from_edit(edit: OperatorEdit) -> List[PreferenceRule]:
    """
    Analyse an operator edit and extract reusable preference rules.
    """
    provider = get_provider()

    diff = _compute_diff(edit.original_draft, edit.edited_draft)
    additions, deletions = _extract_additions_deletions(diff)

    if not diff.strip():
        logger.info("No meaningful diff found — no rules extracted")
        return []

    # Truncate for context limits
    orig_excerpt = edit.original_draft[:1500]
    edit_excerpt = edit.edited_draft[:1500]
    additions_text = "\n".join(additions[:20])
    deletions_text = "\n".join(deletions[:20])

    prompt = RULE_EXTRACTION_PROMPT.format(
        original_excerpt=orig_excerpt,
        edited_excerpt=edit_excerpt,
        diff=diff[:1000],
        additions=additions_text or "(none)",
        deletions=deletions_text or "(none)",
    )

    try:
        raw = await provider.generate(
            prompt=prompt,
            system_prompt=RULE_EXTRACTION_SYSTEM,
            max_tokens=1024,
            temperature=0.1,
        )
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        rule_data = json.loads(raw)

        new_rules = []
        for rd in rule_data:
            if rd.get("confidence", 0) >= 0.6:
                rule = PreferenceRule(
                    rule_text=rd["rule_text"],
                    rule_category=rd.get("rule_category", "general"),
                    source_edit_ids=[edit.edit_id],
                    confidence=rd.get("confidence", 0.7),
                )
                new_rules.append(rule)

        logger.info(f"Extracted {len(new_rules)} rules from edit {edit.edit_id}")
        return new_rules

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Rule extraction failed: {e}")
        return []


# ── Rule deduplication ────────────────────────────────────────────────────────

DEDUP_SYSTEM = """You are comparing two preference rules for a legal drafting system.
Determine if they express the same underlying preference.
Return ONLY 'yes' or 'no'."""


async def _are_rules_duplicate(rule_a: str, rule_b: str) -> bool:
    """Use LLM to check if two rules are semantically equivalent."""
    provider = get_provider()
    try:
        result = await provider.generate(
            prompt=f"Rule A: {rule_a}\nRule B: {rule_b}\n\nAre these the same preference? (yes/no)",
            system_prompt=DEDUP_SYSTEM,
            max_tokens=5,
            temperature=0.0,
        )
        return result.strip().lower().startswith("yes")
    except Exception:
        # Fallback: simple string similarity
        return rule_a.lower()[:50] == rule_b.lower()[:50]


async def deduplicate_rules(
    existing: List[PreferenceRule],
    new_rules: List[PreferenceRule],
) -> List[PreferenceRule]:
    """
    Merge new rules into existing, deduplicating semantically equivalent ones.
    When a duplicate is found, boost the confidence of the existing rule.
    """
    merged = list(existing)

    for new_rule in new_rules:
        is_dup = False
        for existing_rule in merged:
            if await _are_rules_duplicate(new_rule.rule_text, existing_rule.rule_text):
                # Boost confidence and add source edit
                existing_rule.confidence = min(
                    1.0, existing_rule.confidence + 0.05
                )
                existing_rule.source_edit_ids.extend(new_rule.source_edit_ids)
                is_dup = True
                logger.debug(f"Duplicate rule merged: {new_rule.rule_text[:60]}")
                break
        if not is_dup:
            merged.append(new_rule)

    return merged


# ── Main entry point ──────────────────────────────────────────────────────────

async def process_operator_edit(edit: OperatorEdit) -> List[PreferenceRule]:
    """
    Full improvement loop:
    1. Extract rules from the edit
    2. Deduplicate against existing rules
    3. Save updated rule store
    4. Return newly added/updated rules
    """
    logger.info(f"Processing operator edit: {edit.edit_id}")

    # Extract new rules
    new_rules = await extract_rules_from_edit(edit)

    if not new_rules:
        return []

    # Load existing and deduplicate
    existing_rules = load_preference_rules()
    merged_rules = await deduplicate_rules(existing_rules, new_rules)

    # Save
    save_preference_rules(merged_rules)

    # Return only the genuinely new rules
    existing_ids = {r.rule_id for r in existing_rules}
    truly_new = [r for r in merged_rules if r.rule_id not in existing_ids]

    logger.info(
        f"Edit processed: {len(new_rules)} extracted, "
        f"{len(truly_new)} new rules added, "
        f"{len(merged_rules)} total rules"
    )
    return truly_new
