import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure sys.path includes current working directory, script directory, and src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for d in [os.getcwd(), script_dir, parent_dir, os.path.join(parent_dir, 'src')]:
    if os.path.exists(d) and d not in sys.path:
        sys.path.insert(0, d)

from feature_extraction import extract_url_features, FEATURE_COLUMNS

def predict_single_url(url_string, model_path='models/phishing_pipeline.joblib'):
    """
    Takes a raw URL string, extracts numerical features, loads the trained pipeline,
    and returns prediction probability, classification verdict, and key risk indicators.
    """
    if not os.path.exists(model_path):
        # Search possible model locations
        possible_paths = [
            model_path,
            os.path.join(script_dir, 'models', 'phishing_pipeline.joblib'),
            os.path.join(parent_dir, 'models', 'phishing_pipeline.joblib'),
            'models/phishing_pipeline.joblib'
        ]
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        if found_path:
            model_path = found_path
        else:
            raise FileNotFoundError(
                f"Trained model pipeline not found at '{model_path}'. "
                "Please run train_classifier.py first to train and export the model."
            )

    # 1. Load trained pipeline (includes scaler + model)
    pipeline = joblib.load(model_path)

    # 2. Wrap URL string in DataFrame
    raw_df = pd.DataFrame([{'url': url_string.strip()}])

    # 3. Extract numerical feature row
    feature_df = extract_url_features(raw_df, url_column='url')
    
    # Isolate feature columns matching model training
    X_input = feature_df[['url'] + FEATURE_COLUMNS]

    # 4. Predict probability and binary class
    prediction = int(pipeline.predict(X_input)[0])
    probabilities = pipeline.predict_proba(X_input)[0]
    
    phishing_prob = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)
    trust_score = float(round((1.0 - phishing_prob) * 100, 2))
    risk_score = float(round(phishing_prob * 100, 2))

    # 5. Extract diagnostic risk triggers
    row = X_input[FEATURE_COLUMNS].iloc[0]
    risk_flags = []
    if row.get('is_ip_address', 0) == 1:
        risk_flags.append("Raw IPv4 Address in Domain")
    if row.get('is_shortened', 0) == 1:
        risk_flags.append("Known URL Shortener Domain")
    if row.get('count_suspicious_keywords', 0) > 0:
        risk_flags.append(f"Suspicious Security Keywords Detected ({int(row.get('count_suspicious_keywords'))})")
    if row.get('domain_has_prefix_suffix_hyphen', 0) == 1:
        risk_flags.append("Domain Prefix/Suffix Hyphen Deception")
    if row.get('domain_entropy', 0) > 3.8:
        risk_flags.append(f"High Domain Randomness / Entropy ({row.get('domain_entropy'):.2f})")
    if row.get('count_at', 0) > 0:
        risk_flags.append("Embedded User Authentication Symbol (@)")
    if row.get('risky_tld', 0) == 1:
        risk_flags.append("Frequently Abused Top-Level Domain")
    if row.get('has_punycode_or_non_ascii', 0) == 1:
        risk_flags.append("Look-alike (Punycode / Non-ASCII) Characters in Domain")
    if row.get('brand_outside_domain', 0) == 1:
        risk_flags.append("Brand Name Used Outside the Real Domain")

    verdict = "PHISHING / MALICIOUS" if prediction == 1 else "LEGITIMATE / SAFE"

    return {
        'url': url_string,
        'verdict': verdict,
        'is_phishing': bool(prediction == 1),
        'risk_score_pct': risk_score,
        'trust_score_pct': trust_score,
        'risk_flags': risk_flags if risk_flags else ["No immediate heuristic threat flags triggered"],
        'features': X_input[FEATURE_COLUMNS].to_dict(orient='records')[0]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict security verdict for a raw URL string.")
    parser.add_argument('--url', type=str, default="http://login-verify-secure-account.com/update", help="Raw URL string to evaluate")
    parser.add_argument('--model', type=str, default="models/phishing_pipeline.joblib", help="Path to trained pipeline")
    
    args = parser.parse_args()

    print("\n--- PHISHING URL DETECTION INFERENCE ENGINE ---")
    print(f"Target URL: {args.url}\n")

    try:
        result = predict_single_url(args.url, model_path=args.model)
        
        print("=== EVALUATION RESULT ===")
        print(f"Verdict:         {result['verdict']}")
        print(f"Risk Score:      {result['risk_score_pct']}% (Phishing Likelihood)")
        print(f"Trust Score:     {result['trust_score_pct']}% (Safety Rating)")
        print("\nDiagnostic Risk Flags:")
        for flag in result['risk_flags']:
            print(f" - [!] {flag}")

        print("\nExtracted Features Breakdown:")
        for k, v in result['features'].items():
            print(f"  * {k:<32}: {v}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
