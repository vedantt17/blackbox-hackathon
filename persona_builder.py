"""
persona_builder.py  —  v2: Raw Q&A persona

Core philosophy change from v1:
  Instead of collapsing 709 answers into 26 aggregate scores and then into
  low/moderate/high labels, we now feed the LLM actual question-answer pairs
  grouped by topic. This preserves signal that scoring throws away.

Structure of each persona:
  1. Demographics (verbatim labels)
  2. Personality snapshot (computed Big Five scores + selected raw BFI items)
  3. Cognitive style (CRT score + selected NFC items verbatim)
  4. Values & social orientation (individualism items, regulatory focus items)
  5. Risk & money behavior (trust game verbatim, risk lottery choices, spendthrift)
  6. Emotional profile (Beck Anxiety score, Beck Depression score, closure items)
  7. Behavioral biases (verbatim bias question outcomes)
  8. Political & worldview (political items verbatim)

Item selection strategy:
  - For long scales (18+ items): pick the ~6 most discriminating items
    (highest variance across the 233-person sample = most informative)
  - For short scales (<8 items): include all
  - For economic games: include verbatim (rich behavioral signal)
  - For bias questions: include verbatim outcome + brief interpretation
"""

import json
import os
import numpy as np
import pandas as pd
from data_loader import load_survey, load_all_surveys

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_q(text: str) -> str:
    """Strip leading boilerplate from question text, clean newlines."""
    text = text.replace("\n", " ").strip()
    # Remove repeated preamble
    for prefix in [
        "Please indicate your agreement with each of the following statements about yourself. - ",
        "Please indicate your agreement with each of the following\nstatements about yourself. - ",
        "Here are a number of characteristics that may or may not apply to you. Please indicate next to each statement the extent to which you agree or disagree with that statement.\n\nI see myself as someone who... - ",
        "Following are a number of characteristics that may or may not apply to you. Please indicate next to each statement the extent to which you agree or di",
        "Please indicate your agreement with each of the following statements about yourself. - ",
        "Following are a number of characteristics that may or may not apply to you. ",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
        # Also try after stripping the full preamble when embedded mid-string
        if " - " in text:
            parts = text.split(" - ", 1)
            if len(parts[0]) > 60:  # long preamble
                text = parts[1]
    return text.strip()


def _top_variance_cols(cols: list, df: pd.DataFrame, n: int) -> list:
    """Return the n columns with highest variance across all persons."""
    variances = {}
    for c in cols:
        if c in df.columns:
            numeric = pd.to_numeric(df[c], errors="coerce")
            variances[c] = float(numeric.var())
    sorted_cols = sorted(variances, key=variances.get, reverse=True)
    return sorted_cols[:n]


def _safe_get(row: pd.Series, col: str) -> str:
    val = row.get(col, None)
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s if s and s != "nan" else None


# ── BFI scoring (same as feature_engineer) ───────────────────────────────────

BFI_REVERSE = {2,6,8,9,12,18,21,23,24,27,31,34,35,37,41,43}
BFI_TRAITS = {
    "Extraversion":      [1,6,11,16,21,26,31,36],
    "Agreeableness":     [2,7,12,17,22,27,32,37,42],
    "Conscientiousness": [3,8,13,18,23,28,33,38,43],
    "Neuroticism":       [4,9,14,19,24,29,34,39],
    "Openness":          [5,10,15,20,25,30,35,40,41,44],
}

def _big_five_scores(row_num: pd.Series) -> dict:
    scores = {}
    for trait, items in BFI_TRAITS.items():
        vals = []
        for i in items:
            col = f"Big Five _{i}"
            try:
                v = float(row_num.get(col, np.nan))
                if not np.isnan(v):
                    vals.append((6 - v) if i in BFI_REVERSE else v)
            except (ValueError, TypeError):
                pass
        scores[trait] = round(float(np.mean(vals)), 2) if vals else None
    return scores


def _crt_score(row_num: pd.Series, row_lab: pd.Series) -> int:
    correct = 0
    if str(row_lab.get("CRT 1", "")).lower().strip() == "emily":
        correct += 1
    if row_num.get("CRT 2") == 0:
        correct += 1
    if row_num.get("CRT 3") == 2:
        correct += 1
    if row_num.get("CRT4") == 8:
        correct += 1
    return correct


# ── Section builders ──────────────────────────────────────────────────────────

def _section_demographics(row_lab: pd.Series) -> list:
    fields = [
        ("Q12", "Sex"), ("Q13", "Age"), ("Q14", "Education"),
        ("Q15", "Race/ethnicity"), ("Q11", "Region"),
        ("Q21", "Household income"), ("Q17", "Marital status"),
        ("Q24", "Employment"), ("Q18", "Religion"),
        ("Q20", "Politics"), ("Q22", "Political views"),
    ]
    lines = ["DEMOGRAPHICS"]
    for col, label in fields:
        v = _safe_get(row_lab, col)
        if v:
            lines.append(f"  {label}: {v}")
    return lines


def _section_big_five(row_num: pd.Series, row_lab: pd.Series,
                       s1_lab_all: pd.DataFrame, q1: dict) -> list:
    scores = _big_five_scores(row_num)
    lines = ["PERSONALITY (Big Five, 1-5 scale)"]
    desc = {
        "Extraversion":      ("introverted/reserved", "moderately social", "outgoing/energetic"),
        "Agreeableness":     ("skeptical/competitive", "cooperative but assertive", "trusting/warm/conflict-averse"),
        "Conscientiousness": ("spontaneous/flexible", "moderately organized", "disciplined/reliable/goal-oriented"),
        "Neuroticism":       ("emotionally stable/calm", "occasionally anxious", "emotionally reactive/anxious"),
        "Openness":          ("conventional/practical", "selectively curious", "imaginative/curious/novelty-seeking"),
    }
    for trait, (low, mid, high) in desc.items():
        s = scores.get(trait)
        if s is None:
            continue
        level = low if s < 2.5 else (high if s > 3.5 else mid)
        lines.append(f"  {trait}: {s}/5 — {level}")

    # Add 3 most discriminating BFI items verbatim (highest variance)
    bf_cols = [f"Big Five _{i}" for i in range(1, 45) if f"Big Five _{i}" in s1_lab_all.columns]
    top3 = _top_variance_cols(bf_cols, s1_lab_all, 3)
    if top3:
        lines.append("  Selected items (actual answers):")
        for col in top3:
            v = _safe_get(row_lab, col)
            q = _clean_q(q1.get(col, col))
            if v and q:
                lines.append(f'    "I see myself as someone who {q.lower()}" → {v}')
    return lines


def _section_cognitive(row_num: pd.Series, row_lab: pd.Series,
                        s1_lab_all: pd.DataFrame, q1: dict) -> list:
    crt = _crt_score(row_num, row_lab)
    lines = ["COGNITIVE STYLE"]
    lines.append(f"  CRT score: {crt}/4 ({'analytical thinker' if crt >= 3 else 'intuitive thinker' if crt <= 1 else 'mixed'})")

    # Top 5 variance NFC items verbatim
    nfc_cols = [c for c in s1_lab_all.columns if "need for cognition" in c.lower()]
    top5 = _top_variance_cols(nfc_cols, s1_lab_all, 5)
    if top5:
        lines.append("  Need for Cognition (selected items):")
        for col in top5:
            v = _safe_get(row_lab, col)
            q = _clean_q(q1.get(col, col))
            if v and q:
                lines.append(f'    "{q}" → {v}')
    return lines


def _section_values(row_lab: pd.Series, s2_lab_all: pd.DataFrame,
                     s3_lab_all: pd.DataFrame, q2: dict, q3: dict) -> list:
    lines = ["VALUES & SOCIAL ORIENTATION"]

    # Top 4 individualism items
    ind_cols = [c for c in s2_lab_all.columns if "Individualism" in c]
    top4 = _top_variance_cols(ind_cols, s2_lab_all, 4)
    if top4:
        lines.append("  Individualism vs collectivism (selected items):")
        for col in top4:
            v = _safe_get(row_lab, col)
            q = _clean_q(q2.get(col, col))
            if v and q:
                lines.append(f'    "{q}" → {v}')

    # All regulatory focus items (only 10, highly informative)
    reg_cols = [c for c in s3_lab_all.columns if "Regulatory Focus" in c]
    if reg_cols:
        lines.append("  Regulatory focus (promotion = goals/gains, prevention = safety/obligations):")
        for col in reg_cols[:6]:
            v = _safe_get(row_lab, col)
            q = _clean_q(q3.get(col, col))
            if v and q:
                lines.append(f'    "{q}" → {v}')
    return lines


def _section_risk_money(row_lab: pd.Series, row_num: pd.Series,
                         s2_lab_all: pd.DataFrame, q2: dict) -> list:
    lines = ["RISK & MONEY BEHAVIOR"]

    # Trust game — include verbatim, it's pure behavioral signal
    trust_sender = [c for c in s2_lab_all.columns if "Trust game - sender" in c]
    trust_receiver = [c for c in s2_lab_all.columns if "Trust - receiver" in c]

    for col in trust_sender[:1]:
        v = _safe_get(row_lab, col)
        q = _clean_q(q2.get(col, col))
        if v:
            lines.append(f"  Trust game (sender): {v}")

    for col in trust_receiver[:3]:
        v = _safe_get(row_lab, col)
        q = _clean_q(q2.get(col, col))
        if v and q:
            lines.append(f"  Trust game (receiver, {q[-50:]!r}): {v}")

    # Ultimatum game
    ult_cols = [c for c in s2_lab_all.columns if "Ultimatum" in c]
    for col in ult_cols[:2]:
        v = _safe_get(row_lab, col)
        q = _clean_q(q2.get(col, col))
        if v and q:
            lines.append(f"  Ultimatum game ({q[-50:]}): {v}")

    # Risk lotteries — summarize as tendency rather than all 40
    risk_cols = [c for c in s2_lab_all.columns if "Risk Aversion" in c]
    risk_num_cols = [c for c in row_num.index if "Risk Aversion" in c]
    if risk_num_cols:
        # 1=safe, 2=lottery — proportion choosing lottery = risk-seeking
        vals = pd.to_numeric(row_num[risk_num_cols], errors="coerce").dropna()
        if len(vals):
            lottery_pct = (vals == 2).mean()
            tendency = "risk-seeking" if lottery_pct > 0.6 else ("risk-averse" if lottery_pct < 0.4 else "risk-neutral")
            lines.append(f"  Risk lottery choices: chose risky option {lottery_pct:.0%} of the time ({tendency})")

    # Loss aversion — same approach
    loss_cols = [c for c in row_num.index if "Loss Aversion" in c]
    if loss_cols:
        vals = pd.to_numeric(row_num[loss_cols], errors="coerce").dropna()
        if len(vals):
            loss_pct = (vals == 2).mean()
            lines.append(f"  Loss aversion choices: chose safe option {loss_pct:.0%} of the time")

    # Spendthrift
    st_cols = [c for c in s2_lab_all.columns if "Spendthrift" in c]
    for col in st_cols[:2]:
        v = _safe_get(row_lab, col)
        q = _clean_q(q2.get(col, col))
        if v and q:
            lines.append(f'  Spendthrift: "{q[-80:]}" → {v}')

    # Financial literacy score
    fl_cols = [c for c in row_num.index if "Financial literacy" in c]
    if fl_cols:
        vals = pd.to_numeric(row_num[fl_cols], errors="coerce").dropna()
        if len(vals):
            lines.append(f"  Financial literacy: {vals.mean():.2f}/5 avg ({len(fl_cols)} items)")

    return lines


def _section_emotional(row_lab: pd.Series, row_num: pd.Series,
                        s2_lab_all: pd.DataFrame, s3_lab_all: pd.DataFrame,
                        q2: dict, q3: dict) -> list:
    lines = ["EMOTIONAL PROFILE"]

    # Beck Anxiety — sum score with interpretation
    anx_cols = [c for c in row_num.index if "Beck Anxiety" in c]
    if anx_cols:
        vals = pd.to_numeric(row_num[anx_cols], errors="coerce").dropna()
        if len(vals):
            total = vals.sum()
            level = "minimal" if total < 8 else ("mild" if total < 16 else ("moderate" if total < 26 else "severe"))
            lines.append(f"  Anxiety (Beck): {total:.0f} — {level}")

    # Beck Depression
    dep_cols = [c for c in row_num.index if "Beck Depression" in c]
    if dep_cols:
        vals = pd.to_numeric(row_num[dep_cols], errors="coerce").dropna()
        if len(vals):
            total = vals.sum()
            level = "minimal" if total < 10 else ("mild" if total < 19 else ("moderate" if total < 29 else "severe"))
            lines.append(f"  Depression (Beck): {total:.0f} — {level}")

    # Need for closure — top 4 items verbatim
    nfc_cols = [c for c in s3_lab_all.columns if "Need for closure" in c]
    top4 = _top_variance_cols(nfc_cols, s3_lab_all, 4)
    if top4:
        lines.append("  Need for closure (selected items):")
        for col in top4:
            v = _safe_get(row_lab, col)
            q = _clean_q(q3.get(col, col))
            if v and q:
                lines.append(f'    "{q}" → {v}')

    # Maximization
    max_cols = [c for c in s3_lab_all.columns if "Maximization" in c]
    top3 = _top_variance_cols(max_cols, s3_lab_all, 3)
    if top3:
        lines.append("  Maximization tendency (selected items):")
        for col in top3:
            v = _safe_get(row_lab, col)
            q = _clean_q(q3.get(col, col))
            if v and q:
                lines.append(f'    "{q}" → {v}')
    return lines


def _section_biases(row_lab: pd.Series, q4: dict) -> list:
    lines = ["BEHAVIORAL BIASES (actual answers)"]

    bias_items = [
        ("Form A _1",    "Base rate (Jack/engineer problem, correct=30%)", None),
        ("outcome_bias", "Outcome bias",                                    None),
        ("sunk_cost",    "Sunk cost fallacy",                               {"yes": "susceptible", "no": "resistant"}),
        ("Allais",       "Allais paradox",                                  {"1": "consistent EU", "2": "Allais violator"}),
        ("disease",      "Disease framing (gain vs loss frame)",            None),
        ("linda",        "Linda problem (conjunction fallacy)",
                         {"conjunction": "susceptible to conjunction fallacy",
                          "no_conjunction": "avoids conjunction fallacy"}),
        ("african",      "African countries % (anchoring)",                None),
        ("redwood",      "Redwood height % (anchoring)",                   None),
        ("absolute",     "Absolute vs relative savings (jacket/calculator)", None),
        ("Myside",       "Myside bias",                                     None),
        ("less_is_more", "Less-is-more effect",                             None),
        ("Thaler",       "Thaler mental accounting",                        None),
        ("matching",     "Matching/preference reversal",                    None),
    ]

    for col, label, interp in bias_items:
        if col not in row_lab.index:
            continue
        v = _safe_get(row_lab, col)
        if not v:
            continue
        note = interp.get(v.lower(), "") if interp else ""
        suffix = f" ({note})" if note else ""
        lines.append(f"  {label}: {v}{suffix}")

    return lines


def _section_political(row_lab: pd.Series, s1_lab_all: pd.DataFrame, q1: dict) -> list:
    lines = ["POLITICAL & WORLDVIEW"]

    # False consensus self items (what % they think share their views)
    fc_self_cols = [c for c in s1_lab_all.columns if "False Cons. self" in c or "False cons. self" in c]
    top4 = _top_variance_cols(fc_self_cols, s1_lab_all, 4)
    if top4:
        lines.append("  Policy support estimates (what % of public they think agrees):")
        for col in top4:
            v = _safe_get(row_lab, col)
            q = _clean_q(q1.get(col, col))
            if v and q:
                lines.append(f'    "{q[-70:]}" → estimates {v}%')

    # GREEN scale (environmental attitudes)
    green_cols = [c for c in s1_lab_all.columns if "GREEN" in c]
    for col in green_cols[:3]:
        v = _safe_get(row_lab, col)
        q = _clean_q(q1.get(col, col))
        if v and q:
            lines.append(f'  Environmental attitude: "{q[-70:]}" → {v}')

    return lines


# ── Master builder ────────────────────────────────────────────────────────────

def build_persona(person_id: str,
                  row_lab: pd.Series, row_num: pd.Series,
                  s1_lab_all: pd.DataFrame, s2_lab_all: pd.DataFrame,
                  s3_lab_all: pd.DataFrame,
                  q1: dict, q2: dict, q3: dict, q4: dict) -> str:
    sections = [
        f"=== DIGITAL TWIN: {person_id} ===",
        "",
    ]

    for section_fn, args in [
        (_section_demographics,  (row_lab,)),
        (_section_big_five,      (row_num, row_lab, s1_lab_all, q1)),
        (_section_cognitive,     (row_num, row_lab, s1_lab_all, q1)),
        (_section_values,        (row_lab, s2_lab_all, s3_lab_all, q2, q3)),
        (_section_risk_money,    (row_lab, row_num, s2_lab_all, q2)),
        (_section_emotional,     (row_lab, row_num, s2_lab_all, s3_lab_all, q2, q3)),
        (_section_biases,        (row_lab, q4)),
        (_section_political,     (row_lab, s1_lab_all, q1)),
    ]:
        try:
            lines = section_fn(*args)
            if len(lines) > 1:  # skip empty sections
                sections.extend(lines)
                sections.append("")
        except Exception as e:
            sections.append(f"  [section error: {e}]")
            sections.append("")

    sections.append("=== END ===")
    return "\n".join(sections)


def build_all_personas() -> dict:
    print("[persona_builder] Loading surveys...")
    s1_lab, q1 = load_survey(1, use_labels=True)
    s1_num, _  = load_survey(1, use_labels=False)
    s2_lab, q2 = load_survey(2, use_labels=True)
    s2_num, _  = load_survey(2, use_labels=False)
    s3_lab, q3 = load_survey(3, use_labels=True)
    s3_num, _  = load_survey(3, use_labels=False)
    s4_lab, q4 = load_survey(4, use_labels=True)

    # Merge numeric surveys for per-person numeric access
    num_merged = s1_num.join(s2_num, how="outer", lsuffix="", rsuffix="_s2")
    num_merged = num_merged.join(s3_num, how="outer", lsuffix="", rsuffix="_s3")
    num_merged = num_merged.join(s4_lab, how="outer", lsuffix="", rsuffix="_s4")

    # Merge label surveys
    lab_merged = s1_lab.join(s2_lab, how="outer", lsuffix="", rsuffix="_s2")
    lab_merged = lab_merged.join(s3_lab, how="outer", lsuffix="", rsuffix="_s3")
    lab_merged = lab_merged.join(s4_lab, how="outer", lsuffix="", rsuffix="_s4")

    all_ids = s1_lab.index.tolist()
    personas = {}

    for i, pid in enumerate(all_ids, 1):
        print(f"  [{i}/{len(all_ids)}] Building persona for {pid}...", end="\r")

        row_lab = lab_merged.loc[pid] if pid in lab_merged.index else pd.Series(dtype=object)
        row_num = num_merged.loc[pid] if pid in num_merged.index else pd.Series(dtype=object)

        # Use s4_lab directly for bias answers
        row_s4 = s4_lab.loc[pid] if pid in s4_lab.index else pd.Series(dtype=object)
        # Merge s4 into row_lab for bias section
        combined_lab = row_lab.copy()
        for col in s4_lab.columns:
            if col not in combined_lab.index:
                combined_lab[col] = row_s4.get(col, np.nan)

        personas[pid] = build_persona(
            pid, combined_lab, row_num,
            s1_lab, s2_lab, s3_lab,
            q1, q2, q3, q4
        )

    print(f"\n[persona_builder] Built {len(personas)} personas")
    return personas


def save_personas(personas: dict):
    path = os.path.join(CACHE_DIR, "personas.json")
    with open(path, "w") as f:
        json.dump(personas, f, indent=2)
    print(f"[persona_builder] Saved to {path}")


def load_personas() -> dict:
    path = os.path.join(CACHE_DIR, "personas.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    personas = build_all_personas()
    sample = list(personas.keys())[0]
    print(f"\n--- Sample persona ({sample}) ---")
    print(personas[sample])
    save_personas(personas)