import json
import traceback
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def get_recommendations(anomaly_report):

    recommendations = {}

    for col, stats in anomaly_report.items():

        if client is None:
            recommendation, reason = _rule_based_recommendation(col, stats)
        else:
            try:
                recommendation, reason = _gemini_recommendation(col, stats)
                print(f"[GEMINI] {col}: {recommendation}")
            except Exception as e:
                print(f"[FALLBACK] {col}: {e}")
                traceback.print_exc()
                recommendation, reason = _rule_based_recommendation(col, stats)

        recommendations[col] = {
            "recommendation": recommendation,
            "reason": reason
        }

    return recommendations


def get_negative_recommendations(negative_report):
    """
    Generates AI recommendations per column with negative values.
    One recommendation per column, not per row.
    """

    if not negative_report.get("columns"):
        return {}

    recommendations = {}

    for col, stats in negative_report["columns"].items():

        if client is None:
            recommendation, reason = _rule_based_negative(col, stats)
        else:
            try:
                recommendation, reason = _gemini_negative_column(col, stats)
                print(f"[GEMINI NEG] {col}: {recommendation}")
            except Exception as e:
                print(f"[FALLBACK NEG] {col}: {e}")
                traceback.print_exc()
                recommendation, reason = _rule_based_negative(col, stats)

        recommendations[col] = {
            "recommendation": recommendation,
            "reason": reason
        }

    return recommendations


def _gemini_recommendation(col, stats):

    prompt = f"""
You are an AI assistant for a telecom data preprocessing system.

A column named '{col}' has been analyzed for anomalies.

Statistics:
- Total anomalies detected: {stats['count']}
- Percentage of dataset: {stats['percentage']}%
- IQR Lower Limit: {stats['lower_limit']}
- IQR Upper Limit: {stats['upper_limit']}

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
    "recommendation": "Retain Records" or "Adjust Values" or "Exclude Records",
    "reason": "One sentence explanation specific to telecom context"
}}

Rules:
- If anomaly percentage is less than 1%, recommend Retain Records
- If anomaly percentage is between 1% and 5%, recommend Adjust Values
- If anomaly percentage is greater than 5%, recommend Adjust Values
- Reason should mention the column name and telecom context
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
    recommendation = parsed["recommendation"]
    reason = parsed["reason"]

    valid = ["Retain Records", "Adjust Values", "Exclude Records"]
    if recommendation not in valid:
        raise ValueError(f"Invalid recommendation: {recommendation}")

    return recommendation, reason


def _gemini_negative_column(col, stats):

    prompt = f"""
You are an AI assistant for a telecom data preprocessing system.

A column named '{col}' has {stats['count']} negative values
({stats['percentage']}% of the dataset).

The column median (for positive values) is {stats['median_value']}.

In telecom data, negative values in numeric fields are generally invalid.
However, the best correction depends on the column context.

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
    "recommendation": "Replace with Median" or "Replace with 0" or "Delete Rows",
    "reason": "One sentence explanation specific to this column in telecom context"
}}

Rules:
- For charge/cost/revenue columns (MonthlyCharges, TotalCharges): recommend Replace with Median
- For usage/count columns: recommend Replace with 0
- If column percentage exceeds 5%: recommend Delete Rows as data is unreliable
- Reason should be specific to the column name
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
    recommendation = parsed["recommendation"]
    reason = parsed["reason"]

    valid = ["Replace with Median", "Replace with 0", "Delete Rows"]
    if recommendation not in valid:
        raise ValueError(f"Invalid recommendation: {recommendation}")

    return recommendation, reason


def _rule_based_recommendation(col, stats):

    if stats["percentage"] < 1.0:
        return (
            "Retain Records",
            "Very few anomalies detected. Likely valid business records."
        )

    return (
        "Adjust Values",
        "Anomaly percentage is significant. Capping to IQR limits recommended."
    )


def _rule_based_negative(col, stats):

    if stats["column_threshold_exceeded"]:
        return (
            "Delete Rows",
            "High percentage of negative values suggests unreliable column data."
        )

    col_lower = col.lower()
    if any(word in col_lower for word in ["charge", "cost", "revenue", "amount"]):
        return (
            "Replace with Median",
            "Negative charges are invalid in telecom billing. Replacing with median preserves distribution."
        )

    return (
        "Replace with 0",
        "Negative values replaced with zero as a safe default for this column type."
    )