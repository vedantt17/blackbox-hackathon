"""
run.py  —  BlackBox Hackathon pipeline (v2: LLM-only, rich personas)

Approach B (ML similarity) dropped — it was generating noise not signal.
Focus is entirely on Approach A: rich raw Q&A personas + LLM prediction.

Usage:
  python run.py --stage build     # Run once NOW — builds and caches all 233 personas
  python run.py --stage predict   # Run when test questions drop (fill in TEST_QUESTIONS)
  python run.py --stage test      # Smoke test — no API calls needed
  python run.py --stage eval      # Internal cross-val on held-out survey questions

Fill in TEST_QUESTIONS below when the test set is revealed.
"""

import argparse
import json
import os
import pandas as pd
from datetime import datetime

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILL THIS IN WHEN TEST QUESTIONS ARE REVEALED (last 4 hours)
TEST_QUESTIONS = [
    # {
    #     "question": "...",
    #     "options": ["opt1", "opt2", ...],   # or None if open-ended/numeric
    # },
]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def stage_build():
    print("=" * 60)
    print("STAGE: BUILD")
    print("=" * 60)

    from persona_builder import build_all_personas, save_personas
    from data_loader import load_all_surveys
    from ml_pipeline import MLPipeline

    print("Building rich Q&A personas for all 233 people...")
    personas = build_all_personas()
    save_personas(personas)
    avg_chars = sum(len(p) for p in personas.values()) / len(personas)
    print(f"  Avg persona: {avg_chars:.0f} chars (~{avg_chars/4:.0f} tokens)")

    print()
    print("Training ML pipeline (297 models)...")
    raw_num, qmap = load_all_surveys(use_labels=False)
    raw_lab, _    = load_all_surveys(use_labels=True)
    ml = MLPipeline()
    ml.fit(raw_num, raw_lab, qmap)
    ml.save()

    print()
    print("BUILD COMPLETE")
    print("  LLM personas: cache/personas.json")
    print("  ML pipeline:  cache/ml_pipeline.pkl")


def stage_predict():
    print("=" * 60)
    print("STAGE: PREDICT (ML + LLM hybrid)")
    print("=" * 60)
    import glob

    from persona_builder import load_personas
    from predictor import predict_single
    from ml_pipeline import MLPipeline

    # ── Find input JSON ──────────────────────────────────────────
    json_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.json"))
    json_files = [f for f in json_files if "cache" not in f and "personas" not in f]
    if not json_files:
        print("ERROR: No test JSON found. Put your test questions JSON in the project folder.")
        return
    test_file = sorted(json_files)[-1]
    print(f"Test file: {test_file}")

    with open(test_file) as f:
        questions = json.load(f)
    print(f"Found {len(questions)} question-person pairs")

    # ── Load both approaches ─────────────────────────────────────
    print("Loading LLM personas...")
    personas = load_personas()

    print("Loading ML pipeline...")
    if MLPipeline.is_cached():
        ml = MLPipeline.load()
    else:
        print("  ML not cached — run --stage build first")
        ml = None

    # ── Routing threshold ────────────────────────────────────────
    ML_THRESHOLD = 0.08   # similarity score above which we trust ML

    def clean_options(opts):
        if isinstance(opts, str):
            return None
        return [str(o).strip() for o in opts]

    def parse_numeric_range(opts_str):
        parts = str(opts_str).replace("to", " ").split()
        nums = [int(p) for p in parts if p.strip().lstrip("-").isdigit()]
        return (nums[0], nums[-1]) if len(nums) >= 2 else (0, 100)

    def build_full_question(q):
        parts = []
        if q.get("context"):
            parts.append(f"Context: {q['context']}")
        parts.append(q["question_text"])
        return "  ".join(parts)

    # ── Predict ──────────────────────────────────────────────────
    results = []
    ml_count  = 0
    llm_count = 0
    total = len(questions)

    for i, q in enumerate(questions, 1):
        pid    = q["person_id"]
        q_id   = q["question_id"]
        full_q = build_full_question(q)
        opts   = clean_options(q["options"])
        is_numeric = opts is None

        print(f"[{i}/{total}] {q_id} | {full_q[:55]}")

        if pid not in personas:
            print(f"  WARNING: {pid} not in persona cache")
            q["predicted_answer"] = None
            results.append(q)
            continue

        predicted_answer = None
        approach_used    = "none"

        if is_numeric:
            # Numeric range question — ML with regression fallback
            scale_range = parse_numeric_range(q["options"])
            if ml:
                res = ml.predict_numeric(pid, full_q, scale_range)
                predicted_answer = res["predicted_answer"]
                approach_used = "ML-numeric"
            else:
                mid = (scale_range[0] + scale_range[1]) // 2
                predicted_answer = str(mid)
                approach_used = "fallback-midpoint"

        else:
            # Option question — check ML confidence first
            ml_conf = ml.get_confidence_for_question(full_q) if ml else 0.0

            if ml and ml_conf >= ML_THRESHOLD:
                # ML path
                res = ml.predict(pid, full_q, opts)
                predicted_answer = res.get("predicted_answer")
                approach_used = f"ML (sim={ml_conf:.2f})"
                ml_count += 1
            else:
                # LLM path
                res = predict_single(pid, personas[pid], full_q, opts)
                pred_num  = res.get("predicted_option_number")
                pred_text = res.get("predicted_answer", "")
                if pred_num and isinstance(pred_num, int) and 1 <= pred_num <= len(opts):
                    predicted_answer = opts[pred_num - 1]
                elif pred_text in opts:
                    predicted_answer = pred_text
                else:
                    pred_lower = str(pred_text).lower()
                    for opt in opts:
                        opt_text = opt.split(" - ", 1)[-1].lower() if " - " in opt else opt.lower()
                        if opt_text in pred_lower or pred_lower in opt_text:
                            predicted_answer = opt
                            break
                    if not predicted_answer:
                        predicted_answer = pred_text
                approach_used = f"LLM (sim={ml_conf:.2f})"
                llm_count += 1

        q["predicted_answer"] = predicted_answer

        # Convert to option number for API submission
        # API expects: integer option number (1-indexed) or 0-100 for numeric
        if is_numeric:
            try:
                q["predicted_answer_num"] = int(predicted_answer)
            except (ValueError, TypeError):
                q["predicted_answer_num"] = 50
        else:
            # Extract number from option string e.g. "3 - Backpacking" -> 3
            # or find position in options list
            try:
                opts = clean_options(q.get("options", []))
                if opts and predicted_answer in opts:
                    q["predicted_answer_num"] = opts.index(predicted_answer) + 1
                elif predicted_answer and str(predicted_answer)[0].isdigit():
                    q["predicted_answer_num"] = int(str(predicted_answer).split(" - ")[0].strip())
                else:
                    q["predicted_answer_num"] = 1
            except (ValueError, TypeError, AttributeError):
                q["predicted_answer_num"] = 1

        print(f"  -> {predicted_answer} (#{q['predicted_answer_num']})  [{approach_used}]")
        results.append(q)

    # ── Save ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = os.path.join(OUTPUTS_DIR, f"predictions_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Save submission-ready JSON (what the API expects)
    submission = [
        {
            "person_id":        r["person_id"],
            "question_id":      r["question_id"],
            "predicted_answer": r.get("predicted_answer_num", 1),
        }
        for r in results
    ]
    sub_path = os.path.join(OUTPUTS_DIR, f"submission_{ts}.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"  Submission JSON: {sub_path}")

    out_csv = os.path.join(OUTPUTS_DIR, f"predictions_{ts}.csv")
    pd.DataFrame([{
        "person_id":        r["person_id"],
        "question_id":      r["question_id"],
        "question_text":    r["question_text"],
        "predicted_answer": r["predicted_answer"],
        "predicted_answer_num": r.get("predicted_answer_num"),
    } for r in results]).to_csv(out_csv, index=False)

    print()
    print("PREDICTIONS COMPLETE")
    print(f"  ML used:  {ml_count}/{total} questions")
    print(f"  LLM used: {llm_count}/{total} questions")
    print(f"  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")


def stage_test():
    """Smoke test — no API calls. Verifies persona quality only."""
    print("=" * 60)
    print("STAGE: TEST")
    print("=" * 60)

    from persona_builder import load_personas
    from predictor import build_prediction_prompt

    print("\n[1/2] Loading cached personas...")
    try:
        personas = load_personas()
    except FileNotFoundError:
        print("  No cache found. Run --stage build first.")
        return

    print(f"  ✓ {len(personas)} personas loaded")

    sample_id = list(personas.keys())[0]
    persona_text = personas[sample_id]
    print(f"\n[2/2] Sample persona ({sample_id}):")
    print("  " + "\n  ".join(persona_text.split("\n")[:25]))
    print(f"  ... ({len(persona_text)} chars total)")

    test_q    = "How likely are you to share personal data with a company for a discount?"
    test_opts = ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"]
    prompt = build_prediction_prompt(persona_text, test_q, test_opts)
    print(f"\n  LLM prompt would be {len(prompt)} chars ({len(prompt)//4} tokens)")
    print("\n✓ ALL CHECKS PASSED")
    print("Run --stage eval to check accuracy on held-out questions (needs API key)")


def stage_eval():
    """
    Honest cross-validation using scales completely absent from the persona.
    Need for Uniqueness and Self-Monitoring are not included in persona text,
    so the LLM cannot memorize answers. This gives real accuracy numbers.
    """
    print("=" * 60)
    print("STAGE: EVAL (honest holdout cross-validation)")
    print("=" * 60)

    from persona_builder import load_personas
    from predictor import predict_single
    from data_loader import load_survey

    print("Loading cached personas...")
    personas = load_personas()

    s3_lab, _ = load_survey(3, use_labels=True)
    s3_num, _ = load_survey(3, use_labels=False)

    # These items are completely absent from persona text - true holdout
    EVAL_ITEMS = [
        {
            "col": "Need for uniqueness _1",
            "question": "I often combine possessions in such a way that I create a personal image that cannot be duplicated.",
            "options": ["Disagree strongly", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree strongly"],
        },
        {
            "col": "Need for uniqueness _5",
            "question": "When it comes to the products I buy and situations I use them, I have broken customs and rules.",
            "options": ["Disagree strongly", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree strongly"],
        },
        {
            "col": "Need for uniqueness _9",
            "question": "When a product I own becomes popular among the general population, I begin to use it less.",
            "options": ["Disagree strongly", "Disagree a little", "Neither agree nor disagree", "Agree a little", "Agree strongly"],
        },
        {
            "col": "Self-Monitoring_1",
            "question": "In social situations, I have the ability to alter my behavior if I feel that something else is called for.",
            "options": ["Certainly, always false", "Generally false", "Somewhat false, but with exceptions", "Somewhat true, but with exceptions", "Generally true", "Certainly, always true"],
        },
        {
            "col": "Self-Monitoring_5",
            "question": "I have found that I can adjust my behavior to meet the requirements of any situation I find myself in.",
            "options": ["Certainly, always false", "Generally false", "Somewhat false, but with exceptions", "Somewhat true, but with exceptions", "Generally true", "Certainly, always true"],
        },
    ]

    all_scores = []

    for item in EVAL_ITEMS:
        col      = item["col"]
        question = item["question"]
        options  = item["options"]

        if col not in s3_lab.columns:
            print(f"  Skipping {col} not found")
            continue

        print()
        print(f"Eval Q: {question[:80]}")

        # Ground truth: actual label from survey
        ground_truth = {}
        for pid in personas:
            if pid in s3_lab.index:
                val = str(s3_lab.loc[pid, col]).strip()
                if val and val != "nan" and val in options:
                    ground_truth[pid] = val

        if not ground_truth:
            print(f"  No ground truth found for {col}")
            continue

        # Majority baseline
        from collections import Counter
        most_common = Counter(ground_truth.values()).most_common(1)[0]
        majority_acc = most_common[1] / len(ground_truth)
        chance_acc   = 1 / len(options)

        print(f"  Persons with ground truth: {len(ground_truth)}")
        print(f"  Majority baseline: {majority_acc:.1%} (always predict {most_common[0]!r})")
        print(f"  Chance baseline:   {chance_acc:.1%}")

        # Sample 30 people for eval to stay under Groq rate limits
        import random
        random.seed(42)
        sample_pids = random.sample(list(ground_truth.keys()), min(30, len(ground_truth)))
        subset = {pid: personas[pid] for pid in sample_pids}
        sample_truth = {pid: ground_truth[pid] for pid in sample_pids}
        ground_truth = sample_truth  # use sample for scoring
        print(f"  Running {len(subset)} LLM predictions (sampled for rate limits)...")

        from predictor import predict_all
        results = predict_all(subset, question, options)

        # Print first 5 raw predictions so we can see what format LLM returns
        print("  Sample raw predictions (first 5):")
        for r in results[:5]:
            reasoning = r.get("reasoning", "")[:60] if r.get("error") else ""
            print(f"    answer={r.get('predicted_answer')!r:35} num={r.get('predicted_option_number')} {reasoning}")

        correct = 0
        total   = 0
        for r in results:
            pid       = r["person_id"]
            true_ans  = ground_truth.get(pid)
            pred_text = str(r.get("predicted_answer") or "").strip()
            pred_num  = r.get("predicted_option_number")
            if not true_ans:
                continue
            total += 1
            matched = False
            # 1. Option number match (most reliable)
            if pred_num and isinstance(pred_num, int) and 1 <= pred_num <= len(options):
                if options[pred_num - 1] == true_ans:
                    matched = True
            # 2. Exact string match
            if not matched and pred_text == true_ans:
                matched = True
            # 3. Strip number prefix e.g. "1 - Disagree strongly" -> "Disagree strongly"
            if not matched:
                clean_pred = pred_text.split(" - ", 1)[-1].strip().lower()
                if clean_pred == true_ans.lower():
                    matched = True
            if matched:
                correct += 1

        acc = correct / total if total else 0
        lift_vs_majority = acc - majority_acc
        lift_vs_chance   = acc - chance_acc
        print(f"  Accuracy: {correct}/{total} = {acc:.1%}")
        print(f"  Lift vs majority: {lift_vs_majority:+.1%}")
        print(f"  Lift vs chance:   {lift_vs_chance:+.1%}")
        all_scores.append(acc)

    if all_scores:
        mean_acc = sum(all_scores) / len(all_scores)
        print()
        print("EVAL COMPLETE")
        print(f"  Questions evaluated: {len(all_scores)}")
        print(f"  Mean accuracy: {mean_acc:.1%}")
        print(f"  These are TRUE holdout questions not seen in persona")


def stage_submit():
    """Submit latest predictions to scoring API."""
    import glob
    print("=" * 60)
    print("STAGE: SUBMIT")
    print("=" * 60)

    TEAM_TOKEN = os.environ.get("TEAM_TOKEN", "")
    if not TEAM_TOKEN:
        print("ERROR: Set your team token first:")
        print("  $env:TEAM_TOKEN=\"your_token_here\"")
        return

    # Find latest submission JSON
    sub_files = glob.glob(os.path.join(OUTPUTS_DIR, "submission_*.json"))
    if not sub_files:
        print("ERROR: No submission file found. Run --stage predict first.")
        return
    sub_file = sorted(sub_files)[-1]
    print(f"Submitting: {sub_file}")

    with open(sub_file) as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} predictions")

    # Preview first 3
    print("  Preview:")
    for p in predictions[:3]:
        print(f"    {p}")

    confirm = input("\nSubmit? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    import requests as req
    resp = req.post(
        "https://blackboxhackathon-production.up.railway.app/score",
        headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
        json=predictions,
    )
    if resp.status_code == 200:
        print("\nSCORE REPORT:")
        print(resp.text)
    elif resp.status_code == 401:
        print("ERROR: Invalid team token.")
    elif resp.status_code == 429:
        print("ERROR: Submission limit reached.")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlackBox Hackathon Pipeline")
    parser.add_argument("--stage", choices=["build", "predict", "test", "eval", "submit"],
                        default="test")
    args = parser.parse_args()

    {"build":   stage_build,
     "predict": stage_predict,
     "test":    stage_test,
     "eval":    stage_eval,
     "submit":  stage_submit}[args.stage]()