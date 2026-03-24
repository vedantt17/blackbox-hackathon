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


# ── Direct demographic answer lookup ─────────────────────────────────────────
# For questions that map directly to known survey columns, bypass ML/LLM
# and return the answer we already know from the person's survey data.

def _load_demographic_lookup():
    """Load person -> demographic answers mapping."""
    from data_loader import load_survey
    s1l, _ = load_survey(1, use_labels=True)
    lookup = {}
    for pid in s1l.index:
        lookup[pid] = {
            'employment': str(s1l.loc[pid, 'Q24']).strip() if 'Q24' in s1l.columns else None,
            'education':  str(s1l.loc[pid, 'Q14']).strip() if 'Q14' in s1l.columns else None,
            'politics':   str(s1l.loc[pid, 'Q20']).strip() if 'Q20' in s1l.columns else None,
            'pol_views':  str(s1l.loc[pid, 'Q22']).strip() if 'Q22' in s1l.columns else None,
            'religion':   str(s1l.loc[pid, 'Q18']).strip() if 'Q18' in s1l.columns else None,
            'church':     str(s1l.loc[pid, 'Q19']).strip() if 'Q19' in s1l.columns else None,
            'income':     str(s1l.loc[pid, 'Q21']).strip() if 'Q21' in s1l.columns else None,
            'region':     str(s1l.loc[pid, 'Q11']).strip() if 'Q11' in s1l.columns else None,
            'race':       str(s1l.loc[pid, 'Q15']).strip() if 'Q15' in s1l.columns else None,
            'age':        str(s1l.loc[pid, 'Q13']).strip() if 'Q13' in s1l.columns else None,
            'edu':        str(s1l.loc[pid, 'Q14']).strip() if 'Q14' in s1l.columns else None,
            'marital':    str(s1l.loc[pid, 'Q17']).strip() if 'Q17' in s1l.columns else None,
        }

    return lookup


def _match_employment_to_options(emp_status, options):
    """Map Q24 employment to T5-style options."""
    emp = emp_status.lower() if emp_status else ''
    opts_lower = [o.lower() for o in options]

    # Direct label matches first
    for i, opt in enumerate(options):
        if emp in opt.lower() or opt.lower().rstrip('.').strip() in emp:
            return i + 1

    # Semantic mapping
    if any(w in emp for w in ['full-time', 'part-time', 'self-employed']):
        for i, opt in enumerate(opts_lower):
            if 'yes' in opt or 'work' in opt:
                return i + 1
    if 'retired' in emp:
        for i, opt in enumerate(opts_lower):
            if 'retired' in opt:
                return i + 1
    if any(w in emp for w in ['unemployed', 'home-maker', 'student']):
        for i, opt in enumerate(opts_lower):
            if 'no' in opt and len(opt) < 10:
                return i + 1
    return None


def _match_politics_to_voting(politics, pol_views, options):
    """Infer 2020 voting from party affiliation."""
    pol = (politics or '').lower()
    views = (pol_views or '').lower()
    opts_lower = [o.lower() for o in options]

    # Strong signals
    if 'democrat' in pol or 'liberal' in views:
        for i, opt in enumerate(opts_lower):
            if 'biden' in opt:
                return i + 1
    if 'republican' in pol or 'conservative' in views:
        for i, opt in enumerate(opts_lower):
            if 'trump' in opt:
                return i + 1
    if 'independent' in pol or 'moderate' in views:
        for i, opt in enumerate(opts_lower):
            if 'other' in opt:
                return i + 1
    # Check for voted/not voted options
    for i, opt in enumerate(opts_lower):
        if 'voted' in opt and 'not' not in opt and 'did' not in opt:
            return i + 1
    return None


DEMOGRAPHIC_QUESTION_KEYWORDS = {
    'employment':    ['work for', 'work last week', 'employment status', 'do any work'],
    'voting_who':    ['joe biden', 'donald trump', 'who did you vote for', 'vote for president'],
    'voting_did':    ['did you vote', 'vote in 2020', 'voted for president'],
    'trust_press':   ['press', 'news media', 'newspapers', 'trust in the press'],
    'affirmative':   ['affirmative action', 'past discrimination', 'black people should be given'],
    'gov_vs_ind':    ['government should', 'people should take care of themselves', 'living standards'],
    'financial_sat': ['getting along financially', 'financial situation', 'standard of living',
                      'pretty well satisfied', 'family income', 'financially these days'],
    'discrimination':['poorer service', 'treated worse', 'unfair treatment', 'discriminated'],
    'own_rent':      ['own your home', 'pay rent', 'buying', 'paying rent'],
    'school':        ['enrolled in a high school', 'enrolled in a college', 'enrolled in a university'],
    'health':        ['your own health', 'health, in general', 'excellent, good, fair'],
    'internet':      ['use the internet', 'internet on any', 'past 12 months'],
    'father_edu':    ["father completed", "father's education", "highest level of education your father"],
    'volunteer':     ['volunteer activities', 'activities for which', 'not paid'],
}

def _detect_demographic_type(question_text):
    """Detect if a question maps to a known demographic answer."""
    q_lower = question_text.lower()
    for dtype, keywords in DEMOGRAPHIC_QUESTION_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return dtype
    return None


def _load_political_lookup():
    """Load person -> political opinion scores for direct prediction."""
    from data_loader import load_all_surveys
    raw_num, _ = load_all_surveys(use_labels=False)
    raw_lab, _ = load_all_surveys(use_labels=True)

    lookup = {}
    fc_cols = [c for c in raw_lab.columns if 'False Cons. self' in c]

    for pid in raw_lab.index:
        # Q22: 1=very conservative ... 5=very liberal
        q22_raw = raw_num.loc[pid, 'Q22'] if 'Q22' in raw_num.columns else None
        try:
            ideology = float(q22_raw)  # 1-5 scale
        except (TypeError, ValueError):
            ideology = 3.0  # default moderate

        q20 = str(raw_lab.loc[pid, 'Q20']).strip() if 'Q20' in raw_lab.columns else ''

        # Political opinion items (False Consensus scale)
        opinions = {}
        for col in fc_cols:
            val = str(raw_lab.loc[pid, col]).strip()
            if val and val != 'nan':
                opinions[col] = val

        nfc_cols = [c for c in raw_num.columns if 'need for cognition' in c.lower()]
        agentic_cols = [c for c in raw_num.columns if 'Agentic Communal' in c]
        sm_cols = [c for c in raw_num.columns if 'Self-Monitoring' in c]
        # Openness items (Big Five standard: 5,10,15,20,25,30,35,40,44)
        openness_cols = [c for c in raw_num.columns if c in
                        ['Big Five _5','Big Five _10','Big Five _15','Big Five _20',
                         'Big Five _25','Big Five _30','Big Five _35','Big Five _40','Big Five _44']]

        import numpy as np
        try:
            nfc_val = float(pd.to_numeric(raw_num.loc[pid, nfc_cols], errors='coerce').mean())
        except:
            nfc_val = 3.0
        try:
            comm_val = float(pd.to_numeric(raw_num.loc[pid, agentic_cols], errors='coerce').mean())
        except:
            comm_val = 5.5
        try:
            sm_val = float(pd.to_numeric(raw_num.loc[pid, sm_cols], errors='coerce').mean())
        except:
            sm_val = 5.0
        try:
            open_val = float(pd.to_numeric(raw_num.loc[pid, openness_cols], errors='coerce').mean())
        except:
            open_val = 3.5

        lookup[pid] = {
            'ideology': ideology,
            'party': q20,
            'opinions': opinions,
            'nfc_raw': nfc_val,
            'sm_raw': sm_val,
            'openness_raw': open_val,
            'sharing_raw': (nfc_val + comm_val) / 2,
        }

    # Add False Consensus direct opinions
    fc_map = {
        'fc4':  'False Cons. self _4',
        'fc6':  'False Cons. self _6',
        'fc7':  'False Cons. self _7',
        'fc9':  'False Cons. self _9',
        'fc1':  'False Cons. self _1',
    }
    fc_support = {
        'Strongly support': 5, 'Somewhat support': 4,
        'Neither oppose nor support': 3,
        'Somewhat oppose': 2, 'Strongly oppose': 1,
    }
    for pid in lookup:
        for key, col in fc_map.items():
            if col in raw_lab.columns and pid in raw_lab.index:
                val = str(raw_lab.loc[pid, col]).strip()
                lookup[pid][key] = fc_support.get(val, 3)
            else:
                lookup[pid][key] = 3

    # Normalize all scores to 1-7 range across all people
    def normalize(key, lo=1, hi=7):
        vals = [v[key] for v in lookup.values()]
        mn, mx = min(vals), max(vals)
        for pid in lookup:
            raw = lookup[pid][key]
            lookup[pid][key + '_norm'] = (raw - mn) / (mx - mn) * (hi - lo) + lo if mx > mn else (lo + hi) / 2

    normalize('sharing_raw')   # sharing_raw_norm: 1-7
    normalize('nfc_raw')       # nfc_raw_norm: 1-7
    normalize('sm_raw')        # sm_raw_norm: 1-7
    normalize('openness_raw')  # openness_raw_norm: 1-7

    # Ideology strength (distance from moderate)
    all_ideo = [v['ideology'] for v in lookup.values()]
    for pid in lookup:
        ideo = lookup[pid]['ideology']
        strength = abs(ideo - 3)
        lookup[pid]['ideo_strength_norm'] = strength / 2 * 6 + 1  # 1-7

    # Sharing propensity 1-5 (for backward compat)
    for pid in lookup:
        lookup[pid]['sharing_propensity'] = (lookup[pid]['sharing_raw_norm'] - 1) / 6 * 4 + 1

    return lookup


def _predict_political_opinion(pid, pol_data, question_text, options):
    """
    Use ideology score + party to predict political opinion questions.
    ideology: 1=very conservative, 5=very liberal
    Returns option number (1-indexed) or None if unsure.
    """
    if not pol_data:
        return None

    ideology = pol_data.get('ideology', 3.0)
    party = pol_data.get('party', '').lower()
    opinions = pol_data.get('opinions', {})
    q_lower = question_text.lower()
    opts_lower = [o.lower() for o in options]
    n_opts = len(options)

    def find_opt(keywords, opts):
        for i, opt in enumerate(opts):
            if any(kw in opt for kw in keywords):
                return i + 1
        return None

    # ── Affirmative action / racial equity ──────────────────────────
    if any(w in q_lower for w in ['affirmative action', 'past discrimination', 'black people should be given',
                                   'racial', 'discrimination']):
        # Calibrated from False Consensus proxy data (path to citizenship item)
        # ideology 5 (very liberal)   -> Strongly favor (86% of this group)
        # ideology 4 (liberal)        -> Strongly favor (49%) or Favor (40%)
        # ideology 3 (moderate)       -> Favor (27%) slight plurality
        # ideology 2 (conservative)   -> Strongly oppose (31%) plurality
        # ideology 1 (very conserv.)  -> Strongly oppose (52%) clear majority
        if ideology >= 4.0:
            return find_opt(['strongly favor', 'strongly support'], opts_lower) or find_opt(['favor', 'support'], opts_lower)
        elif ideology >= 3.0:
            return find_opt(['favor', 'support'], opts_lower)
        else:
            return find_opt(['strongly oppose'], opts_lower) or find_opt(['oppose'], opts_lower)

    # ── Government vs individual responsibility ──────────────────────
    if any(w in q_lower for w in ['government should', 'people should take care of themselves',
                                   'living standards', 'individual responsibility', 'government help',
                                   'standard of living of all poor', 'washington should do everything',
                                   'government in washington', 'each person should take care',
                                   'point 1', 'point 5']):
        # Liberal -> government should help
        # Conservative -> people should take care of themselves
        if ideology >= 4.0:
            return find_opt(['government should improve', 'government', 'strongly agree the government'], opts_lower) or 1
        elif ideology <= 2.0:
            return find_opt(['take care of themselves', 'people should', 'strongly agree that people'], opts_lower)
        else:
            return find_opt(['agree with both', 'both', 'neither'], opts_lower) or ((n_opts + 1) // 2)

    # ── Trust RATINGS T77-T84 (0-100) — BEFORE institution trust ────────
    if 'trustworthy' in q_lower:
        TRUST_MAP = {
            'bbc':               {1: 38, 2: 45, 3: 55, 4: 68, 5: 75},
            'pbs':               {1: 48, 2: 55, 3: 62, 4: 72, 5: 78},
            'economist':         {1: 42, 2: 48, 3: 55, 4: 65, 5: 70},
            'wall street':       {1: 65, 2: 62, 3: 55, 4: 48, 5: 42},
            'wsj':               {1: 65, 2: 62, 3: 55, 4: 48, 5: 42},
            'reddit':            {1: 15, 2: 20, 3: 28, 4: 33, 5: 35},
            'quora':             {1: 28, 2: 30, 3: 35, 4: 38, 5: 40},
            'national enquirer': {1: 8,  2: 7,  3: 6,  4: 5,  5: 4},
            'enquirer':          {1: 8,  2: 7,  3: 6,  4: 5,  5: 4},
            'funny times':       {1: 12, 2: 10, 3: 9,  4: 8,  5: 7},
        }
        ideo_int = max(1, min(5, round(ideology)))
        for source_key, trust_vals in TRUST_MAP.items():
            if source_key in q_lower:
                return trust_vals[ideo_int]
        return 35

    # ── Trust in institutions ────────────────────────────────────────
    if (any(w in q_lower for w in ['press', 'newspapers', 'congress', 'supreme court',
                                   'confidence in']) or
        ('media' in q_lower and 'social media' not in q_lower) or
        ('trust' in q_lower and 'trustworthy' not in q_lower)):
        # Low trust across all groups but more for conservatives on press, liberals on courts
        if 'press' in q_lower or 'media' in q_lower or 'newspaper' in q_lower:
            # Both sides distrust media but for different reasons — use ideology
            if ideology <= 2.0:
                return find_opt(['hardly any', 'very little', 'none'], opts_lower)
            elif ideology >= 4.0:
                return find_opt(['only some', 'some'], opts_lower)
            else:
                return find_opt(['hardly any', 'only some'], opts_lower)
        # General trust
        if ideology >= 4.0:
            return find_opt(['only some', 'some'], opts_lower)
        else:
            return find_opt(['hardly any', 'very little'], opts_lower)

    # ── Financial satisfaction ───────────────────────────────────────
    if any(w in q_lower for w in ['getting along financially', 'financial', 'money', 'income',
                                   'satisfied with your financial']):
        return None  # let ML handle this

    # ── Immigration attitudes (T16) — use FC6/FC9 direct answers ──────────
    if any(w in q_lower for w in ['immigrants from other countries', 'immigrants should be',
                                   'immigrants take jobs']):
        # FC6 = path to citizenship (pro-immigrant), FC9 = deportations (anti-immigrant)
        fc6 = pol_data.get('fc6', 3)  # 1=oppose citizenship, 5=support citizenship
        fc9 = pol_data.get('fc9', 3)  # 1=oppose deportations, 5=support deportations
        # Combined: pro-immigrant score = fc6 - fc9 + 3 (range 1-5 approx)
        pro_immigrant = max(1, min(5, fc6 - fc9 + 3))
        # T16 asks "immigrants take jobs away" -> agree=anti-immigrant, disagree=pro-immigrant
        # Pro-immigrant (5) -> Strongly disagree (high option number)
        # Anti-immigrant (1) -> Strongly agree (option 1)
        if pro_immigrant >= 4:
            return find_opt(['strongly disagree', 'disagree'], opts_lower) or max(1, n_opts - 1)
        elif pro_immigrant >= 3:
            return find_opt(['disagree', 'neither'], opts_lower)
        elif pro_immigrant <= 2:
            return find_opt(['strongly agree', 'agree'], opts_lower) or 1
        else:
            return find_opt(['neither', 'no opinion'], opts_lower)

    # ── Healthcare / higher taxes (T19) — use FC4 direct answer ────────────
    if any(w in q_lower for w in ['pay higher taxes', 'higher taxes to improve', 'health care for all']):
        fc4 = pol_data.get('fc4', 3)  # 1=oppose Medicare, 5=support Medicare
        # fc4=5 (strongly support Medicare) -> very willing to pay taxes
        # fc4=1 (strongly oppose Medicare) -> very unwilling
        if fc4 >= 5:
            return find_opt(['very willing'], opts_lower) or 1
        elif fc4 >= 4:
            return find_opt(['fairly willing', 'willing'], opts_lower)
        elif fc4 <= 1:
            return find_opt(['very unwilling'], opts_lower)
        elif fc4 <= 2:
            return find_opt(['fairly unwilling', 'unwilling'], opts_lower)
        else:
            return find_opt(['neither', 'neutral'], opts_lower)

    # ── Hard work vs luck (T20) ───────────────────────────────────────────
    if any(w in q_lower for w in ['people get ahead', 'hard work', 'lucky breaks', 'luck or help']):
        if ideology >= 4.0:
            return find_opt(['luck', 'help from other'], opts_lower)
        elif ideology <= 2.0:
            return find_opt(['hard work most important', 'hard work'], opts_lower)
        else:
            return find_opt(['equally', 'both'], opts_lower)

    # ── Trust in people (T21) ─────────────────────────────────────────────
    if any(w in q_lower for w in ['take advantage of you', 'try to be fair', 'most people would']):
        # Higher agreeableness -> more trusting; use ideology as proxy
        if ideology >= 4.0:
            return find_opt(['try to be fair', 'fair'], opts_lower)
        elif ideology <= 2.0:
            return find_opt(['take advantage', 'advantage'], opts_lower)
        else:
            return find_opt(['depends', 'it depends'], opts_lower)

    # ── Data privacy / TikTok (T23/T24) ──────────────────────────────────
    if any(w in q_lower for w in ['data they collect online', 'chinese government', 'tiktok']):
        # Bipartisan concern but conservatives more worried about China specifically
        if 'chinese government' in q_lower or 'tiktok' in q_lower:
            if ideology <= 2.0:
                return find_opt(['strongly agree', 'agree'], opts_lower)
            else:
                return find_opt(['agree', 'somewhat'], opts_lower)
        else:
            # General data privacy: both sides concerned
            return find_opt(['very concerned', 'somewhat concerned'], opts_lower)

    # ── T45-T54: Importance of sharing factors (1-7 scale) ──────────────
    # Detect factor name for T45-T54
    # T45-T49: bare factor name only ("Headline", "Source" etc) -> let ML handle
    # T50-T54: long question ending with factor name -> use personality handlers
    _factor = None
    _q_stripped = q_lower.strip()
    for _f in ['number of likes', 'political lean', 'content type (entertaining vs. informative)',
                'source', 'headline']:
        if _q_stripped.endswith(_f):
            # Only handle if it's a long question (T50-T54), not bare factor (T45-T49)
            if len(_q_stripped) > len(_f) + 10:
                _factor = _f
            break  # break either way - T45-T49 falls through to ML
    if _factor is not None:
        factor = _factor
        # Use ideology distance for political lean — best spread (std=2.15)
        ideo = pol_data.get('ideology', 3.0)
        ideo_dist = abs(ideo - 3.0)  # 0=moderate, 2=very strong
        # Use self-monitoring raw for likes (std=1.27, range 2.3-10.0)
        sm = pol_data.get('sm_raw', 5.0)

        if factor == 'political lean':
            # Strong ideologues care more about political lean (0-2 -> 1-7)
            return max(1, min(7, round(ideo_dist * 3 + 1)))
        elif factor == 'number of likes':
            # High self-monitors care about social proof
            # sm range 2.3-10.0 -> scale to 1-7
            return max(1, min(7, round((sm - 2.3) / 7.7 * 6 + 1)))
        elif factor == 'source':
            # NFC + Openness: intellectually curious people care more about source
            nfc = pol_data.get('nfc_raw_norm', 4.0)
            openness = pol_data.get('openness_raw_norm', 4.0)
            score = (nfc + openness) / 2
            return max(1, min(7, round(score)))
        elif factor == 'headline':
            # Sharers + open people care about headline quality
            sharing = pol_data.get('sharing_raw_norm', 4.0)
            openness = pol_data.get('openness_raw_norm', 4.0)
            return max(1, min(7, round((sharing + openness) / 2)))
        else:
            # Content type: Openness predicts preferring informative content
            openness = pol_data.get('openness_raw_norm', 4.0)
            return max(1, min(7, round(openness)))

    # ── T57: Is this headline funny? ─────────────────────────────────────
    if 'is the above headline funny' in q_lower or 'funny, amusing, or entertaining' in q_lower:
        # Openness + sharing propensity predicts humor appreciation
        openness = pol_data.get('openness_raw_norm', 4.0)
        sharing = pol_data.get('sharing_propensity', 3.0)
        # High openness + high sharing = more likely to find funny
        funny_score = (openness + sharing * 1.5) / 2.5
        # Options: 1=Extremely unfunny ... 7=Extremely funny
        return max(1, min(n_opts, round((funny_score - 1) / 6 * (n_opts - 1) + 1)))

    # ── News sharing likelihood (T27-T76) ───────────────────────────────
    # T27-T44: question text is article metadata (headline/source), options have 'unlikely to share'
    # T57-T73: question text has 'how likely would you be to share'
    is_sharing_q = (
        any(w in q_lower for w in ['how likely', 'likely would you be to share',
                                   'unlikely to share', 'very unlikely to share',
                                   'share this', 'sharing this']) or
        ('headline:' in q_lower and 'source:' in q_lower) or
        any('unlikely to share' in o.lower() for o in options)
    )
    if is_sharing_q:
        sharing = pol_data.get('sharing_propensity', 3.0)
        ideology = pol_data.get('ideology', 3.0)
        if n_opts == 0:
            return None

        # ── Source trust signal (T27-T44 have source embedded in question) ──
        # Person's trust in the source predicts willingness to share from it
        TRUST_MAP = {
            'bbc':      {1: 38, 2: 45, 3: 55, 4: 68, 5: 75},
            'pbs':      {1: 48, 2: 55, 3: 62, 4: 72, 5: 78},
            'reddit':   {1: 15, 2: 20, 3: 28, 4: 33, 5: 35},
            'quora':    {1: 28, 2: 30, 3: 35, 4: 38, 5: 40},
        }
        ideo_int = max(1, min(5, round(ideology)))
        source_trust = None
        for src, trust_vals in TRUST_MAP.items():
            if src in q_lower:
                source_trust = trust_vals[ideo_int]
                break

        # ── Article virality (content-based) ──────────────────────────────
        ARTICLE_VIRALITY = {
            'solid gold toilet': 5, 'scramble': 4,
            'elderly men': 5, 'nursing home': 5, 'heavy metal': 5,
            'sword fight': 5, 'ex-wife': 5, 'kansas man': 4,
            'japanese billionaire': 4, 'girlfriend': 4, 'maezawa': 4,
            'spotify': 4, 'playlists for dogs': 5,
            'teen discovers': 4, 'nasa intern': 4, 'planet': 4,
            'pink bananas': 4, 'cotton candy': 4,
            'seat assignment': 4, 'bathroom': 4,
            'bermuda triangle': 3, 'mystery': 3,
            'singing spider': 3, 'dolphins': 3,
            'gold coin': 3, 'einstein': 3,
            'rat-sized elephants': 4,
            'jurassic park': 4, 'dinosaur dna': 4,
            'glow-in-the-dark': 3,
            'plants that emit': 2, 'tiny-house': 2,
        }
        virality = 3
        for keyword, score in ARTICLE_VIRALITY.items():
            if keyword in q_lower:
                virality = score
                break

        # ── Political lean alignment signal ───────────────────────────────
        pol_lean_score = 0  # neutral
        if 'political lean: liberal' in q_lower:
            # Liberal article: liberal person more likely to share (+1), conservative less (-1)
            pol_lean_score = (ideology - 3) * 0.5  # -1 to +1
        elif 'political lean: conservative' in q_lower:
            # Conservative article: conservative person more likely (+1), liberal less (-1)
            pol_lean_score = (3 - ideology) * 0.5  # -1 to +1

        # ── Number of likes signal ────────────────────────────────────────
        likes_score = 0
        sm_raw = pol_data.get('sm_raw', 5.0)  # self-monitoring raw score
        # High self-monitoring people are more influenced by likes
        sm_sensitivity = (sm_raw - 2.3) / 7.7  # 0-1 scale
        if '2,000 likes' in q_lower or '2000 likes' in q_lower:
            likes_score = sm_sensitivity * 1.5   # high likes boosts sharing for SM people
        elif '200 likes' in q_lower:
            likes_score = sm_sensitivity * 0.5
        elif '20 likes' in q_lower:
            likes_score = -sm_sensitivity * 0.5  # low likes reduces sharing for SM people

        # ── Combine all signals ───────────────────────────────────────────
        if source_trust is not None:
            trust_score = (source_trust - 10) / 70 * 4 + 1  # 0-100 -> 1-5
            combined = (virality * 0.3 + trust_score * 0.3 +
                       sharing * 0.2 + pol_lean_score + likes_score + 3 * 0.2)
        else:
            combined = virality * 0.5 + sharing * 0.3 + pol_lean_score + likes_score + 3 * 0.2

        combined = max(1, min(5, combined))

        # Scale to option range (1=very unlikely ... n=very likely)
        option_idx = round((combined - 1) / 4 * (n_opts - 1))
        option_idx = max(0, min(n_opts - 1, option_idx))
        return option_idx + 1

    # ── T56/T76: Would you ever share news on social media ───────────────
    if 'would you ever consider sharing news' in q_lower:
        sharing = pol_data.get('sharing_propensity', 3.0)
        if sharing >= 3.5:
            return 1  # Yes
        elif sharing <= 1.5:
            return 3  # I don't use social media
        else:
            return 1  # Yes

    # ── T55/T75: Verifiably correct vs Entertaining scale ────────────────
    if 'verifiably correct' in q_lower and 'entertaining' in q_lower:
        # NFC predicts preferring verifiable correctness
        nfc = pol_data.get('nfc_raw_norm', 4.0)
        if nfc >= 5.0:
            return find_opt(['verifiably correct'], opts_lower) or 1
        elif nfc <= 3.0:
            # Find middle or entertaining end
            return min(n_opts, max(1, round(n_opts * 0.7)))
        else:
            return max(1, min(n_opts, round(n_opts / 2)))

    # ── T74: Important to only share verified news ───────────────────────
    if 'important to only share news' in q_lower or 'important to only share verified' in q_lower:
        nfc = pol_data.get('nfc_raw_norm', 4.0)
        if nfc >= 5.0:
            return find_opt(['strongly agree'], opts_lower) or 1
        elif nfc >= 3.5:
            return find_opt(['agree', 'somewhat agree'], opts_lower) or 2
        else:
            return find_opt(['somewhat agree', 'neither'], opts_lower) or 3

    return None


POLITICAL_QUESTION_KEYWORDS = [
    'affirmative action', 'past discrimination', 'black people should be given',
    'government should', 'people should take care of themselves', 'living standards',
    'press', 'newspapers', 'trust', 'confidence in', 'congress', 'supreme court',
    'abortion', 'immigration', 'gun', 'tax', 'healthcare', 'climate',
    'racial', 'discrimination', 'individual responsibility',
    # T18 specific - government vs individual responsibility scale
    'standard of living of all poor', 'washington should do everything',
    'government in washington', 'each person should take care',
    'point 1', 'point 5',  # scale question markers
    # T77-T84 source trust ratings
    'how trustworthy', 'trustworthy is this source', 'trust this source',
    # T15-T24 new political/social questions
    'family life suffer', 'immigrants from other countries',
    'pay higher taxes', 'people get ahead', 'hard work', 'lucky breaks',
    'take advantage of you', 'try to be fair',
    'chinese government', 'tiktok', 'data they collect online',
    # News sharing likelihood T27-T76
    'how likely', 'likely would you be to share', 'unlikely to share',
    'very unlikely to share', 'share this article', 'sharing this',
    # T50-T54 importance ratings (long question versions only - T45-T49 go to ML)
    'to what extent can you use each of these',
    'determine whether a news article is accurate',
    # T55/T75 verifiably correct vs entertaining
    'verifiably correct',
    # T56/T76 would you ever share
    'would you ever consider sharing news',
    # T74 important to only share verified
    'important to only share news', 'important to only share verified',
]

def _is_political_question(question_text):
    q_lower = question_text.lower()
    return any(kw in q_lower for kw in POLITICAL_QUESTION_KEYWORDS)


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

    print("Loading demographic lookup...")
    demo_lookup = _load_demographic_lookup()

    print("Loading political lookup...")
    pol_lookup = _load_political_lookup()

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

        # ── Demographic direct lookup (highest priority) ─────────
        demo_type = _detect_demographic_type(full_q) if not is_numeric else None
        person_demo = demo_lookup.get(pid, {})

        if demo_type and not is_numeric:
            demo_ans_num = None
            if demo_type == "employment":
                emp = person_demo.get("employment")
                if emp:
                    demo_ans_num = _match_employment_to_options(emp, opts)
            elif demo_type == "voting_who":
                pol = person_demo.get("politics")
                views = person_demo.get("pol_views")
                if pol:
                    demo_ans_num = _match_politics_to_voting(pol, views, opts)
            elif demo_type == "voting_did":
                # Everyone in our dataset voted (they're survey respondents)
                for ii, opt in enumerate(opts):
                    if "voted" in opt.lower() and "not" not in opt.lower() and "did" not in opt.lower():
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "financial_sat":
                income = person_demo.get('income', '')
                inc = income.lower()
                if '100,000' in inc:
                    target = 'pretty well satisfied'
                elif '75,000' in inc or '50,000' in inc:
                    target = 'more or less satisfied'
                elif '30,000' in inc or 'less than' in inc:
                    target = 'not satisfied at all'
                else:
                    target = 'more or less satisfied'
                for ii, opt in enumerate(opts):
                    if target in opt.lower():
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "discrimination":
                race = person_demo.get('race', '').lower()
                if any(r in race for r in ['black', 'hispanic', 'asian', 'other']):
                    for ii, opt in enumerate(opts):
                        if any(w in opt.lower() for w in ['once a week', 'few times a month', 'almost every']):
                            demo_ans_num = ii + 1
                            break
                else:
                    for ii, opt in enumerate(opts):
                        if any(w in opt.lower() for w in ['less than once a year', 'never', 'seldom']):
                            demo_ans_num = ii + 1
                            break
            elif demo_type == "own_rent":
                income = person_demo.get('income', '').lower()
                age = person_demo.get('age', '').lower()
                if '100,000' in income or '75,000' in income:
                    target = 'own or is buying'
                elif 'less than $30' in income or '30,000' in income:
                    target = 'paying rent'
                else:
                    target = 'own or is buying'  # majority own
                for ii, opt in enumerate(opts):
                    if target in opt.lower():
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "school":
                age = person_demo.get('age', '').lower()
                if '18-29' in age:
                    target = 'yes'
                else:
                    target = 'no'
                for ii, opt in enumerate(opts):
                    if target in opt.lower() and len(opt) < 6:
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "health":
                age = person_demo.get('age', '').lower()
                income = person_demo.get('income', '').lower()
                if '18-29' in age or '30-49' in age:
                    if '100,000' in income or '75,000' in income:
                        target = 'excellent'
                    else:
                        target = 'good'
                elif '65+' in age:
                    if 'less than $30' in income or '30,000' in income:
                        target = 'fair'
                    else:
                        target = 'good'
                else:
                    target = 'good'
                for ii, opt in enumerate(opts):
                    if target in opt.lower():
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "internet":
                age = person_demo.get('age', '').lower()
                if '18-29' in age or '30-49' in age:
                    target = 'several times a day'
                elif '65+' in age:
                    target = 'once a day'
                else:
                    target = 'several times a day'
                for ii, opt in enumerate(opts):
                    if target in opt.lower():
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "father_edu":
                edu = person_demo.get('edu', '').lower()
                # Father's education correlates with person's education
                if 'postgraduate' in edu or 'college graduate' in edu:
                    target = 'college graduate'
                elif 'some college' in edu or "associate" in edu:
                    target = 'some college'
                elif 'high school' in edu:
                    target = 'high school graduate'
                else:
                    target = 'high school graduate'
                for ii, opt in enumerate(opts):
                    if target in opt.lower():
                        demo_ans_num = ii + 1
                        break
            elif demo_type == "volunteer":
                # Religiosity predicts volunteering
                church = person_demo.get('church', '').lower()
                if any(w in church for w in ['once a week', 'almost every', 'several times']):
                    target = 'yes'
                else:
                    target = 'no'
                for ii, opt in enumerate(opts):
                    if target in opt.lower() and len(opt) < 6:
                        demo_ans_num = ii + 1
                        break

            if demo_ans_num:
                predicted_answer = opts[demo_ans_num - 1]
                approach_used = f"DEMO-{demo_type}"
                q["predicted_answer"] = predicted_answer
                q["predicted_answer_num"] = demo_ans_num
                print(f"  -> {predicted_answer} (#{demo_ans_num})  [{approach_used}]")
                results.append(q)
                continue

        # ── Political direct prediction ──────────────────────────
        if _is_political_question(full_q):
            pol_data = pol_lookup.get(pid, {})
            pol_result = _predict_political_opinion(pid, pol_data, full_q, opts or [])
            if pol_result is not None:
                if is_numeric:
                    # Trust ratings return 0-100 directly
                    predicted_answer = str(int(pol_result))
                    q["predicted_answer"] = predicted_answer
                    q["predicted_answer_num"] = int(pol_result)
                    approach_used = "POLITICAL-numeric"
                    print(f"  -> {predicted_answer}  [{approach_used}]")
                    results.append(q)
                    continue
                elif opts and 1 <= pol_result <= len(opts):
                    predicted_answer = opts[pol_result - 1]
                    approach_used = "POLITICAL"
                    q["predicted_answer"] = predicted_answer
                    q["predicted_answer_num"] = pol_result
                    print(f"  -> {predicted_answer} (#{pol_result})  [{approach_used}]")
                    results.append(q)
                    continue

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