"""
Module 19: Segment-level retention strategy.

One Gemini call per segment (not per customer). Results are cached in
processed/{filename}_retention.json so reopening the page never calls
Gemini again. If Gemini fails for a segment, a rule-based strategy is
used for that segment so the page never breaks.
"""

import os
import json
import traceback

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

VALID_PRIORITY = ["High", "Medium", "Low"]
VALID_CHANNEL = ["Phone call", "SMS", "Email", "In-app offer", "Account manager"]


# ---------------------------------------------------------------------
# 1. Build a statistical profile for each segment
# ---------------------------------------------------------------------
def build_segment_profiles(df, segments):
    """Returns a list of plain-Python dicts, one per segment."""

    total = len(df)
    has_prob = "churn_probability" in df.columns
    profiles = []

    for seg in segments:
        sub = df[df["segment"] == seg]
        n = len(sub)

        if n == 0:
            profiles.append({"segment": seg, "count": 0})
            continue

        churners = sub[sub["predicted_churn"] == "Yes"]
        contract_mix = (
            sub["Contract"].value_counts(normalize=True) * 100
        ).round(1).to_dict()

        profile = {
            "segment": seg,
            "count": int(n),
            "pct_of_customers": round(100 * n / total, 1) if total else 0,
            "predicted_churners": int(len(churners)),
            "churn_rate_pct": round(100 * len(churners) / n, 1),
            "high_risk_pct": round(
                100 * float((sub["risk_level"] == "High").sum()) / n, 1
            ),
            "avg_tenure_months": round(float(sub["tenure"].mean()), 1),
            "avg_monthly_charge": round(float(sub["MonthlyCharges"].mean()), 2),
            "avg_addon_services": (
                round(float(sub["num_services"].mean()), 1)
                if "num_services" in sub.columns else None
            ),
            "contract_mix_pct": {str(k): float(v) for k, v in contract_mix.items()},
            "top_contract": str(sub["Contract"].mode().iloc[0]),
            "revenue_at_risk": round(float(churners["MonthlyCharges"].sum()), 2),
        }
        if has_prob:
            profile["avg_churn_probability_pct"] = round(
                100 * float(sub["churn_probability"].mean()), 1
            )
        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------
# 2. Gemini call (one per segment)
# ---------------------------------------------------------------------
def _gemini_strategy(profile):

    prompt = f"""
You are a customer retention strategist for a telecom company.

A churn-prediction model has grouped customers into behavioural segments.
Here are the statistics for one segment. Base your answer ONLY on these
numbers, do not assume anything else about the segment.

{json.dumps(profile, indent=2)}

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
    "diagnosis": "Two sentences on why this segment is at risk (or not), citing specific numbers above",
    "actions": ["3 to 4 concrete retention actions, each one short sentence"],
    "priority": "High" or "Medium" or "Low",
    "channel": "Phone call" or "SMS" or "Email" or "In-app offer" or "Account manager"
}}

Rules:
- Priority should reflect revenue_at_risk and churn_rate_pct together
- Actions must be specific to this segment's contract mix, tenure and charges
- Do not invent discount percentages larger than 20%
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = json.loads(text)

    diagnosis = str(parsed["diagnosis"]).strip()
    actions = [str(a).strip() for a in parsed["actions"] if str(a).strip()]
    priority = parsed["priority"]
    channel = parsed["channel"]

    if not diagnosis or not (2 <= len(actions) <= 6):
        raise ValueError("Gemini returned an incomplete strategy")
    if priority not in VALID_PRIORITY:
        raise ValueError(f"Invalid priority: {priority}")
    if channel not in VALID_CHANNEL:
        raise ValueError(f"Invalid channel: {channel}")

    return {
        "diagnosis": diagnosis,
        "actions": actions[:4],
        "priority": priority,
        "channel": channel,
    }


# ---------------------------------------------------------------------
# 3. Rule-based fallback (used when Gemini fails or no key is set)
# ---------------------------------------------------------------------
_FALLBACK_ACTIONS = {
    "Premium": [
        "Assign a dedicated account manager for proactive check-ins.",
        "Offer loyalty rewards or a free service upgrade.",
        "Invite them to early access of new plans.",
    ],
    "At-Risk Loyal": [
        "Reach out personally before their contract renewal.",
        "Offer a loyalty discount in exchange for a longer contract.",
        "Ask for feedback and resolve any open service issues.",
    ],
    "Churner - Price Sensitive": [
        "Offer a cheaper plan or a limited-time discount of up to 15%.",
        "Bundle services to lower the effective monthly cost.",
        "Push a switch from month-to-month to a discounted annual plan.",
    ],
    "Churner - Disengaged": [
        "Run a re-engagement campaign showing unused benefits.",
        "Offer a free add-on service trial to increase usage.",
        "Call to understand why usage is low and fix the cause.",
    ],
    "New Customer": [
        "Send a structured onboarding sequence in the first 90 days.",
        "Check in at 30 and 60 days to resolve early problems.",
        "Offer an incentive for moving to a longer contract.",
    ],
    "Stable Low-Value": [
        "Keep contact low-cost with automated email and in-app messages.",
        "Offer relevant add-ons to raise value gradually.",
        "Do not spend on heavy discounts for this segment.",
    ],
}


def _rule_based_strategy(profile):

    seg = profile["segment"]
    churn_rate = profile["churn_rate_pct"]
    rev = profile["revenue_at_risk"]

    if churn_rate >= 50 or rev >= 5000:
        priority = "High"
    elif churn_rate >= 25:
        priority = "Medium"
    else:
        priority = "Low"

    channel = {
        "Premium": "Account manager",
        "At-Risk Loyal": "Phone call",
        "Churner - Price Sensitive": "In-app offer",
        "Churner - Disengaged": "Email",
        "New Customer": "SMS",
        "Stable Low-Value": "Email",
    }.get(seg, "Email")

    diagnosis = (
        f"{profile['predicted_churners']} of {profile['count']} customers "
        f"({churn_rate}%) are predicted to churn, putting about "
        f"{rev:,.0f} of monthly revenue at risk. "
        f"Most are on {profile['top_contract']} contracts with an average "
        f"tenure of {profile['avg_tenure_months']} months."
    )

    return {
        "diagnosis": diagnosis,
        "actions": _FALLBACK_ACTIONS.get(seg, [
            "Contact high-risk customers with a tailored offer.",
            "Review pricing and contract options for this group.",
            "Track churn monthly and adjust the approach.",
        ]),
        "priority": priority,
        "channel": channel,
    }


# ---------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------
def get_retention_strategies(df, segments, cache_path, force=False):
    """
    Returns {"strategies": [...], "gemini_count": int, "fallback_count": int}.
    Loads from cache unless force=True.
    """

    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass  # corrupt cache: regenerate

    profiles = build_segment_profiles(df, segments)
    strategies = []
    gemini_count = 0
    fallback_count = 0

    for profile in profiles:

        if profile["count"] == 0:
            strategies.append({
                "segment": profile["segment"],
                "profile": profile,
                "empty": True,
            })
            continue

        if client is None:
            result = _rule_based_strategy(profile)
            source = "Rule-based"
            fallback_count += 1
        else:
            try:
                result = _gemini_strategy(profile)
                source = "Gemini"
                gemini_count += 1
                print(f"[GEMINI RETENTION] {profile['segment']}: {result['priority']}")
            except Exception as e:
                print(f"[FALLBACK RETENTION] {profile['segment']}: {e}")
                traceback.print_exc()
                result = _rule_based_strategy(profile)
                source = "Rule-based"
                fallback_count += 1

        strategies.append({
            "segment": profile["segment"],
            "profile": profile,
            "empty": False,
            "source": source,
            **result,
        })

    # Biggest revenue at risk first
    strategies.sort(
        key=lambda s: s["profile"].get("revenue_at_risk", 0), reverse=True
    )

    data = {
        "strategies": strategies,
        "gemini_count": gemini_count,
        "fallback_count": fallback_count,
    }

    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data