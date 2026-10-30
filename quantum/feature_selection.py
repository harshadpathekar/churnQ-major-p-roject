import numpy as np
import json
import os
from sklearn.feature_selection import mutual_info_classif

CATEGORY_RULES = [
    # (keywords, category, business rationale). Order matters: first match wins.
    (["tenure"], "Loyalty",
     "newer customers tend to churn more than long-standing ones"),
    (["contract", "paperless", "payment"], "Contract & Billing",
     "contract type and payment setup affect how easily a customer can leave"),
    (["monthly", "total", "charge"], "Spend",
     "billing amount is linked to price-driven churn"),
    (["internet", "phone", "lines", "security", "backup",
      "protection", "techsupport", "tech_support", "streaming"], "Services",
     "subscribed services affect how embedded the customer is"),
    (["gender", "senior", "partner", "dependent"], "Demographics",
     "customer profile can shift churn tendency"),
]


def build_reason(name, score, rank, max_score, selected, n_features):
    lname = name.lower()
    category, rationale = "General", "general customer attribute"
    for keywords, cat, why in CATEGORY_RULES:
        if any(k in lname for k in keywords):
            category, rationale = cat, why
            break

    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.60:
        strength = "Strong"
    elif ratio >= 0.30:
        strength = "Moderate"
    else:
        strength = "Weak"

    if selected:
        return (f"{strength} churn signal (MI {score:.4f}, rank {rank}). "
                f"{category}: {rationale}.")
    return (f"{strength} churn signal (MI {score:.4f}, rank {rank}); "
            f"below the top-{n_features} cutoff, so dropped.")

def select_features(X_train_scaled, y_train, feature_names,
                    filename, processed_folder, n_features=8):
   

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(
        X_train_scaled,
        y_train,
        random_state=42
    )

    # Create score mapping
    feature_scores = sorted(
        zip(feature_names, mi_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top N features
    n_features = min(n_features, len(feature_names))
    top_features = feature_scores[:n_features]
    selected_feature_names = [f[0] for f in top_features]

    # Get indices of selected features
    selected_indices = [
        feature_names.index(name)
        for name in selected_feature_names
    ]

    # Select columns from train set
    X_train_selected = X_train_scaled[:, selected_indices]

    # Save selected feature info
    selection_info = {
        "selected_features": selected_feature_names,
        "selected_indices": selected_indices,
        "n_features": n_features
    }

    selection_path = os.path.join(
        processed_folder,
        f"{filename}_feature_selection.json"
    )
    with open(selection_path, "w") as f:
        json.dump(selection_info, f)

    # Build full report for display
        max_score = float(feature_scores[0][1]) if feature_scores else 0.0

    all_scores = [
        {
            "feature": name,
            "score": round(float(score), 4),
            "selected": name in selected_feature_names,
            "reason": build_reason(
                name, float(score), rank, max_score,
                name in selected_feature_names, n_features
            )
        }
        for rank, (name, score) in enumerate(feature_scores, start=1)
    ]

    selection_report = {
        "total_features_before": len(feature_names),
        "total_features_after": n_features,
        "selected_features": selected_feature_names,
        "all_scores": all_scores
    }

    return X_train_selected, selected_feature_names, selection_report