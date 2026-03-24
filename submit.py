"""
BlackBox Hackathon — Scoring API Starter
==========================================
Use this script to submit your predictions and get your score.

Setup:
    pip install requests

Usage:
    1. Fill in your TEAM_TOKEN below
    2. Load test_questions.json
    3. Fill in predicted_answer for each question
    4. Submit the whole thing — extra fields are ignored
    5. Run: python3 submit.py
"""

import json
import requests

# ── YOUR TEAM TOKEN (replace with your actual token) ──────────────────
TEAM_TOKEN = "TEAM TOKEN"

# ── API URL ───────────────────────────────────────────────────────────
API_URL = "https://blackboxhackathon-production.up.railway.app"


def submit_predictions(predictions):
    """
    Submit predictions and get your score report.

    Just load test_questions.json, fill in predicted_answer, and submit.
    Extra fields (context, options, question_text) are ignored by the API.

    Each prediction needs at minimum:
        - person_id: str
        - question_id: str
        - predicted_answer: number (option number or 0-100)
    """
    resp = requests.post(
        f"{API_URL}/score",
        headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
        json=predictions,
    )


    if resp.status_code == 200:
        print(resp.text)
    elif resp.status_code == 401:
        print("ERROR: Invalid team token. Check your TEAM_TOKEN.")
    elif resp.status_code == 429:
        print("ERROR: Submission limit reached. Contact organizers.")
    else:
        # print(f"ERROR ({resp.status_code}): {resp.text}")
        print(f"ERROR ({resp.status_code}):")


def check_submissions():
    """View your submission history."""
    resp = requests.get(
        f"{API_URL}/submissions",
        headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
    )
    print(resp.text)


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Recommended workflow ──────────────────────────────────────────
    # 1. Load the test questions file
    # with open("test_questions.json") as f:
    #     questions = json.load(f)
    #
    # 2. Fill in predicted_answer using your model
    # for q in questions:
    #     q["predicted_answer"] = your_model_predict(q)
    #
    # 3. Submit the whole thing
    # submit_predictions(questions)

    # ── Or load from your own predictions file ────────────────────────
    # with open("my_predictions.json") as f:
    #     predictions = json.load(f)
    # submit_predictions(predictions)

    # ── Quick test with a single prediction ───────────────────────────
    test = [
        {"person_id": "kficb", "question_id": "T1", "predicted_answer": 3},
    ]

    print("Submitting test prediction...\n")
    submit_predictions(test)

    print("\nSubmission history:\n")
    check_submissions()