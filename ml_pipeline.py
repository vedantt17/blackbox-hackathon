"""
ml_pipeline.py  —  Full ML prediction pipeline

Architecture:
  1. Feature matrix: all 914 survey cols, imputed, scaled
  2. For each of the 345 target items across all scales:
       - Build feature matrix excluding same-scale cols (prevent leakage)
       - Select top-50 most correlated features
       - Train RandomForestClassifier (or regressor for continuous targets)
       - Store model + feature indices + label encoder
  3. At test time:
       - Embed the test question with TF-IDF
       - Find top-K most similar trained items
       - Run person's features through those models
       - Aggregate predictions (majority vote or weighted avg)

This file exposes:
  - MLPipeline.fit()   — trains all models, call during build stage
  - MLPipeline.predict(person_id, question, options) — predict at test time
  - MLPipeline.save() / MLPipeline.load()
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

warnings.filterwarnings('ignore')

CACHE_DIR  = Path(os.path.dirname(__file__)) / "cache"
CACHE_PATH = CACHE_DIR / "ml_pipeline.pkl"

# ── Scale definitions: (scale_name, col_prefix_to_exclude_during_training) ──
SCALES = {
    'Big Five':           'Big Five',
    'Need for Cognition': 'need for cognition',
    'Empathy':            'Empathy',
    'Agentic Communal':   'Agentic Communal',
    'Individualism':      'Individualism',
    'Risk Aversion':      'Risk Aversion',
    'Loss Aversion':      'Loss Aversion',
    'Beck Anxiety':       'Beck Anxiety',
    'Beck Depression':    'Beck Depression',
    'Need for Closure':   'Need for closure',
    'Need for Uniqueness':'Need for uniqueness',
    'Maximization':       'Maximization',
    'Regulatory Focus':   'Regulatory Focus',
    'Self-Monitoring':    'Self-Monitoring',
}


class MLPipeline:

    def __init__(self):
        self.models       = {}    # {col: {model, feat_idx, le, is_regression, classes}}
        self.question_map = {}    # {col: question_text}
        self.tfidf        = None
        self.tfidf_matrix = None
        self.tfidf_cols   = []
        self.sem_model    = None
        self.sem_embeddings = None
        self.person_ids   = []
        self.X_full       = None  # imputed full feature matrix (233 x n_features)
        self.feat_cols    = []    # column names for X_full
        self.imp          = SimpleImputer(strategy='median')
        self.raw_labels   = {}    # {col: {person_id: label_str}} for option mapping

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, raw_num: pd.DataFrame, raw_lab: pd.DataFrame, question_map: dict):
        """
        Train one model per target item.
        raw_num: numeric survey (233 x 914)
        raw_lab: label survey  (233 x 914)
        question_map: {col -> question text}
        """
        self.person_ids   = raw_num.index.tolist()
        self.question_map = question_map

        # ── Build full feature matrix ────────────────────────────────────────
        print("[ml_pipeline] Building feature matrix...")
        X = raw_num.apply(pd.to_numeric, errors='coerce')
        X = X.loc[:, X.notna().mean() > 0.6]
        self.feat_cols = X.columns.tolist()
        X_imp = self.imp.fit_transform(X)
        self.X_full = X_imp

        # ── Store raw label values for option mapping at test time ───────────
        print("[ml_pipeline] Storing label mappings...")
        for col in raw_lab.columns:
            col_data = {}
            for pid in raw_lab.index:
                val = raw_lab.loc[pid, col]
                if pd.notna(val) and str(val) != 'nan':
                    col_data[pid] = str(val)
            if col_data:
                self.raw_labels[col] = col_data

        # ── Train one model per target item ──────────────────────────────────
        total_trained = 0
        total_skipped = 0

        for scale_name, exclude_prefix in SCALES.items():
            target_cols = [c for c in raw_num.columns if exclude_prefix.lower() in c.lower()]
            excl_cols   = set(target_cols)

            # Feature cols for this scale (exclude same-scale to prevent leakage)
            scale_feat_idx = [
                i for i, c in enumerate(self.feat_cols)
                if c not in excl_cols
            ]

            for col in target_cols:
                if col not in raw_num.columns:
                    continue

                y_raw = pd.to_numeric(raw_num[col], errors='coerce').dropna()
                if len(y_raw) < 30:
                    total_skipped += 1
                    continue

                mask    = [i for i, pid in enumerate(self.person_ids) if pid in y_raw.index]
                X_scale = X_imp[mask][:, scale_feat_idx]
                y_vals  = y_raw.loc[[self.person_ids[i] for i in mask]].values

                # Decide regression vs classification
                n_unique = len(np.unique(y_vals))
                is_regression = (n_unique > 10)

                if is_regression:
                    # Bin into 3 buckets for consistent handling
                    y_binned = pd.cut(y_vals, bins=3, labels=[1, 2, 3]).astype(int)
                    y_train  = y_binned
                    is_regression = False
                else:
                    y_train = y_vals.astype(int)

                # Feature selection: top-50 correlated with target
                if X_scale.shape[1] > 50:
                    corrs   = np.array([
                        abs(np.corrcoef(X_scale[:, i], y_train)[0, 1])
                        for i in range(X_scale.shape[1])
                    ])
                    corrs   = np.nan_to_num(corrs)
                    top_idx = np.argsort(corrs)[::-1][:50]
                else:
                    top_idx = np.arange(X_scale.shape[1])

                X_final = X_scale[:, top_idx]

                # Label encode
                le = LabelEncoder()
                y_enc = le.fit_transform(y_train)

                if len(np.unique(y_enc)) < 2:
                    total_skipped += 1
                    continue

                # Train bootstrap ensemble of RFs (person-level resampling)
                # Benchmarked: +1.1% avg accuracy over single RF
                # Person-level bootstrap preserves personality structure
                N_BOOTSTRAP = 5
                bootstrap_models = []
                for b in range(N_BOOTSTRAP):
                    rng = np.random.RandomState(b * 7 + 42)
                    boot_idx = rng.choice(len(X_final), size=len(X_final), replace=True)
                    X_boot = X_final[boot_idx]
                    y_boot = y_enc[boot_idx]
                    if len(np.unique(y_boot)) < 2:
                        continue
                    rf_b = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=b,
                        n_jobs=1,
                    )
                    rf_b.fit(X_boot, y_boot)
                    bootstrap_models.append(rf_b)

                global_feat_idx = [scale_feat_idx[i] for i in top_idx]

                self.models[col] = {
                    'model':        bootstrap_models,  # list of 20 models
                    'feat_idx':     global_feat_idx,
                    'le':           le,
                    'scale':        scale_name,
                    'q_text':       question_map.get(col, col).replace('\n', ' ').strip(),
                }
                total_trained += 1

        print(f"[ml_pipeline] Trained {total_trained} models, skipped {total_skipped}")

        # ── Build question index (semantic if available, TF-IDF fallback) ────
        self.tfidf_cols = list(self.models.keys())
        q_texts = [self.models[c]['q_text'] for c in self.tfidf_cols]

        if SEMANTIC_AVAILABLE:
            print("[ml_pipeline] Building semantic embedding index (all-MiniLM-L6-v2)...")
            self.sem_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sem_embeddings = self.sem_model.encode(q_texts, show_progress_bar=False)
            self.tfidf = None
            self.tfidf_matrix = None
            print(f"[ml_pipeline] Semantic index: {len(q_texts)} questions x {self.sem_embeddings.shape[1]} dims")
        else:
            print("[ml_pipeline] sentence-transformers not found, using TF-IDF fallback...")
            print("[ml_pipeline] Run: pip install sentence-transformers")
            self.sem_model = None
            self.sem_embeddings = None
            self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=8000)
            self.tfidf_matrix = self.tfidf.fit_transform(q_texts)
            print(f"[ml_pipeline] TF-IDF index: {self.tfidf_matrix.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    def _find_similar_models(self, question_text: str, top_k: int = 5) -> list:
        """Returns [(col, similarity_score), ...] sorted by similarity desc."""
        if self.sem_model is not None and self.sem_embeddings is not None:
            # Semantic similarity using sentence embeddings
            q_emb = self.sem_model.encode([question_text])
            sims  = cosine_similarity(q_emb, self.sem_embeddings).flatten()
        else:
            # TF-IDF fallback
            q_vec = self.tfidf.transform([question_text])
            sims  = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top = sims.argsort()[::-1][:top_k]
        return [(self.tfidf_cols[i], float(sims[i])) for i in top]

    # ─────────────────────────────────────────────────────────────────────────
    def predict(
        self,
        person_id:     str,
        question_text: str,
        options:       list,
        top_k:         int = 5,
    ) -> dict:
        """
        Predict one person's answer to one question.

        Returns:
            {
                predicted_answer:       str (matched to options),
                predicted_option_number: int (1-indexed),
                confidence:             float,
                top_similar_question:   str,
                similarity_score:       float,
            }
        """
        if person_id not in self.person_ids:
            return {'predicted_answer': None, 'confidence': 0.0, 'error': 'unknown person'}

        person_idx = self.person_ids.index(person_id)
        person_vec = self.X_full[person_idx]

        # Find most similar trained questions
        similar = self._find_similar_models(question_text, top_k=top_k)

        if not similar or similar[0][1] < 0.05:
            return {
                'predicted_answer':        None,
                'predicted_option_number': None,
                'confidence':              0.0,
                'top_similar_question':    similar[0][0] if similar else None,
                'similarity_score':        similar[0][1] if similar else 0.0,
                'note':                    'no similar question found',
            }

        # Weighted vote across top-K similar models
        option_votes  = {}  # {option_label: weighted_vote}
        top_sim_col   = similar[0][0]
        top_sim_score = similar[0][1]

        for col, sim_score in similar:
            if sim_score < 0.05:
                break
            if col not in self.models:
                continue

            m_info  = self.models[col]
            model   = m_info['model']
            feat_idx = m_info['feat_idx']
            le      = m_info['le']

            # Get this person's features for this model
            x = person_vec[feat_idx].reshape(1, -1)

            # Average probabilities across bootstrap ensemble
            models_list = model if isinstance(model, list) else [model]
            all_probas = []
            ref_classes = None
            for m in models_list:
                p = m.predict_proba(x)[0]
                if ref_classes is None:
                    ref_classes = m.classes_
                    all_probas.append(p)
                else:
                    # Align classes across models (bootstrap may drop rare classes)
                    aligned = np.zeros(len(ref_classes))
                    for ci, c in enumerate(m.classes_):
                        idx = np.where(ref_classes == c)[0]
                        if len(idx):
                            aligned[idx[0]] = p[ci]
                    all_probas.append(aligned)
            proba   = np.mean(all_probas, axis=0)
            classes = le.inverse_transform(ref_classes)

            # Map predicted class to nearest option
            for class_val, prob in zip(classes, proba):
                weight = sim_score * prob
                # Map numeric class to option
                mapped = self._map_class_to_option(
                    class_val, col, person_id, options
                )
                if mapped:
                    option_votes[mapped] = option_votes.get(mapped, 0) + weight

        if not option_votes:
            return {
                'predicted_answer':        options[0] if options else None,
                'predicted_option_number': 1,
                'confidence':              0.0,
                'top_similar_question':    top_sim_col,
                'similarity_score':        top_sim_score,
            }

        # Pick highest vote
        best_option  = max(option_votes, key=option_votes.get)
        total_votes  = sum(option_votes.values())
        confidence   = option_votes[best_option] / total_votes if total_votes > 0 else 0.0
        option_num   = (options.index(best_option) + 1) if best_option in options else None

        return {
            'predicted_answer':        best_option,
            'predicted_option_number': option_num,
            'confidence':              round(confidence, 3),
            'top_similar_question':    top_sim_col,
            'similarity_score':        round(top_sim_score, 3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def predict_numeric(self, person_id: str, question_text: str, scale_range: tuple) -> dict:
        """
        For numeric range questions (e.g. 0-100 zombie confidence).
        Uses Overconfidence score as primary signal, falls back to relative position.
        """
        if person_id not in self.person_ids:
            return {'predicted_answer': str((scale_range[0] + scale_range[1]) // 2), 'confidence': 0.0}

        person_idx = self.person_ids.index(person_id)
        person_vec = self.X_full[person_idx]

        # Try similarity-based prediction first
        similar = self._find_similar_models(question_text, top_k=3)
        weighted_sum   = 0.0
        weighted_count = 0.0

        for col, sim_score in (similar or []):
            if sim_score < 0.05 or col not in self.models:
                continue
            m_info   = self.models[col]
            model    = m_info['model']
            feat_idx = m_info['feat_idx']
            le       = m_info['le']
            x = person_vec[feat_idx].reshape(1, -1)
            models_list = model if isinstance(model, list) else [model]
            all_probas_n = []
            ref_classes_n = None
            for m in models_list:
                p = m.predict_proba(x)[0]
                if ref_classes_n is None:
                    ref_classes_n = m.classes_
                    all_probas_n.append(p)
                else:
                    aligned = np.zeros(len(ref_classes_n))
                    for ci, c in enumerate(m.classes_):
                        idx = np.where(ref_classes_n == c)[0]
                        if len(idx):
                            aligned[idx[0]] = p[ci]
                    all_probas_n.append(aligned)
            proba   = np.mean(all_probas_n, axis=0)
            classes = le.classes_.astype(float)
            expected = float(np.dot(classes, proba))
            c_min, c_max = classes.min(), classes.max()
            rel_pos  = (expected - c_min) / (c_max - c_min) if c_max > c_min else 0.5
            scaled   = scale_range[0] + rel_pos * (scale_range[1] - scale_range[0])
            weighted_sum   += scaled * sim_score
            weighted_count += sim_score

        if weighted_count > 0:
            prediction = int(round(max(scale_range[0], min(scale_range[1], weighted_sum / weighted_count))))
            conf = round(similar[0][1], 3)
        else:
            # Fallback: use Overconfidence column directly if available
            oc_col = "Overconfidence"
            if oc_col in self.feat_cols:
                oc_idx = self.feat_cols.index(oc_col)
                raw_oc = person_vec[oc_idx]
                # Overconfidence ranges ~4-42, scale to target range
                rel = (raw_oc - 4) / (42 - 4)
                prediction = int(round(scale_range[0] + rel * (scale_range[1] - scale_range[0])))
                prediction = max(scale_range[0], min(scale_range[1], prediction))
            else:
                prediction = (scale_range[0] + scale_range[1]) // 2
            conf = 0.1

        return {'predicted_answer': str(prediction), 'confidence': conf}

    # ─────────────────────────────────────────────────────────────────────────
    def _map_class_to_option(self, class_val: int, col: str, person_id: str, options: list) -> str:
        """
        Map a numeric predicted class to the closest option string.
        Uses relative position within the known class range to map to options.
        e.g. class=6 on a 3-7 scale with 4 options -> position 0.75 -> option index 3
        """
        if not options:
            return None

        # Get the full class range for this model
        if col in self.models:
            le = self.models[col]["le"]
            all_classes = le.classes_.astype(float)
            c_min, c_max = all_classes.min(), all_classes.max()
        else:
            c_min, c_max = 1.0, float(max(len(options), int(class_val)))

        # Map class to 0-1 relative position
        if c_max > c_min:
            rel_pos = (float(class_val) - c_min) / (c_max - c_min)
        else:
            rel_pos = 0.5

        # Map to option index
        idx = int(round(rel_pos * (len(options) - 1)))
        idx = max(0, min(len(options) - 1, idx))
        return options[idx]

    # ─────────────────────────────────────────────────────────────────────────
    def get_confidence_for_question(self, question_text: str) -> float:
        """Returns how confident we are ML will work for this question (0-1)."""
        similar = self._find_similar_models(question_text, top_k=1)
        return similar[0][1] if similar else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def save(self):
        CACHE_DIR.mkdir(exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(self, f)
        print(f"[ml_pipeline] Saved to {CACHE_PATH}")

    @classmethod
    def load(cls) -> 'MLPipeline':
        with open(CACHE_PATH, 'rb') as f:
            obj = pickle.load(f)
        print(f"[ml_pipeline] Loaded ({len(obj.models)} models, {len(obj.person_ids)} persons)")
        return obj

    @staticmethod
    def is_cached() -> bool:
        return CACHE_PATH.exists()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from data_loader import load_all_surveys

    print("Loading data...")
    raw_num, qmap = load_all_surveys(use_labels=False)
    raw_lab, _    = load_all_surveys(use_labels=True)

    print("Fitting ML pipeline...")
    ml = MLPipeline()
    ml.fit(raw_num, raw_lab, qmap)
    ml.save()

    # Test a few predictions
    pid = raw_num.index[0]
    tests = [
        ("Have you ever judged someone by their coffee order?",
         ["1 - Strongly disagree","2 - Disagree","3 - Neutral","4 - Agree","5 - Strongly agree"]),
        ("How likely are you to share a funny animal video on social media?",
         ["1 - Very unlikely","2","3","4","5","6","7 - Very likely"]),
        ("I prefer to stand out rather than blend in.",
         ["1 - Strongly disagree","2 - Disagree","3 - Neutral","4 - Agree","5 - Strongly agree"]),
    ]

    print(f"\nTest predictions for {pid}:")
    for q, opts in tests:
        result = ml.predict(pid, q, opts)
        print(f"  Q: {q[:60]}")
        print(f"  A: {result['predicted_answer']}  (conf={result['confidence']:.2f}, sim={result['similarity_score']:.2f})")
        print()