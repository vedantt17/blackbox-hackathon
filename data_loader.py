"""
data_loader.py
Loads all 4 surveys cleanly, handling the 2-row header + ImportId row structure.

CSV structure:
  Row 0: column shortnames (e.g. "Big Five _1", "Q11")
  Row 1: full question text (e.g. "I see myself as someone who... - Is talkative")
  Row 2: ImportId metadata row — NOT a real person, must be skipped
  Row 3+: actual person data (233 people)
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_survey(survey_num: int, use_labels: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Load one survey and return:
      - df: DataFrame with person_id as index, column shortnames as columns
      - question_map: dict of {shortname -> full question text}

    Args:
        survey_num: 1, 2, 3, or 4
        use_labels: if True, load human-readable labels; if False, load numeric codes
    """
    suffix = "labels" if use_labels else "numbers"
    path = os.path.join(DATA_DIR, f"survey_{survey_num}_{suffix}.csv")

    # Read raw — no header parsing yet
    raw = pd.read_csv(path, header=None, dtype=str)

    # Row 0 = shortnames, Row 1 = question texts, Row 2 = ImportId junk, Row 3+ = people
    shortnames = raw.iloc[0].tolist()
    question_texts = raw.iloc[1].tolist()

    # Build question map
    question_map = {
        shortnames[i]: question_texts[i]
        for i in range(len(shortnames))
        if shortnames[i] != "person_id"
    }

    # Build clean DataFrame from row 3 onward
    df = raw.iloc[3:].copy()
    df.columns = shortnames
    df = df.reset_index(drop=True)

    # Set person_id as index
    df = df.set_index("person_id")

    # Convert numeric surveys to float where possible
    if not use_labels:
        df = df.apply(pd.to_numeric, errors="coerce")

    return df, question_map


def load_all_surveys(use_labels: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Load all 4 surveys and merge on person_id.
    Returns merged DataFrame (233 rows x ~918 cols) and combined question map.
    """
    dfs = []
    combined_map = {}

    for i in range(1, 5):
        df, qmap = load_survey(i, use_labels=use_labels)
        dfs.append(df)
        combined_map.update(qmap)

    merged = pd.concat(dfs, axis=1)
    print(f"[data_loader] Loaded {len(merged)} people x {len(merged.columns)} features")
    return merged, combined_map


def get_question_text(col_name: str, question_map: dict) -> str:
    """Helper to get full question text for a column shortname."""
    return question_map.get(col_name, col_name)


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, qmap = load_all_surveys(use_labels=False)
    print(f"Shape: {df.shape}")
    print(f"Sample person IDs: {df.index[:5].tolist()}")
    print(f"Sample columns: {df.columns[:5].tolist()}")
    print(f"\nMissing values per survey (approx):")
    print(df.isnull().sum().sum(), "total NaNs")
