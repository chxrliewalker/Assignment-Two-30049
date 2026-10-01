"""URL feature extraction for ML-based phishing/malicious URL detection."""

import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd


def calculate_entropy(text):
    """Calculates the Shannon entropy of a string (higher means more random)."""
    if not text:
        return 0.0
    counts = Counter(text)
    frequencies = [float(c) / len(text) for c in counts.values()]
    return -sum(f * math.log2(f) for f in frequencies)


def extract_url_features(df, url_column):
    """Transforms raw URL strings into numerical feature columns for ML modeling."""
    # Ensure input is string and lowercase for clean parsing
    urls = df[url_column].astype(str).str.lower().str.strip()

    # 1. Basic URL parsing
    parsed_urls = urls.apply(
        lambda x: urlparse(x) if "://" in x else urlparse("//" + x)
    )
    domains = parsed_urls.apply(lambda x: x.netloc.split(":")[0])  # Strip port
    paths = parsed_urls.apply(lambda x: x.path)

    # 2. Lexical length features
    df["url_len"] = urls.str.len()
    df["domain_len"] = domains.str.len()
    df["path_len"] = paths.str.len()

    # 3. Special character counts
    df["count_dots"] = urls.str.count(r"\.")
    df["count_hyphens"] = urls.str.count("-")
    df["count_slashes"] = urls.str.count("/")
    df["count_questions"] = urls.str.count(r"\?")
    df["count_equal"] = urls.str.count("=")
    df["count_at"] = urls.str.count("@")
    df["count_digits"] = urls.str.count(r"\d")

    # Subdomain depth indicator
    df["domain_count_dots"] = domains.str.count(r"\.")

    # 4. Structural and security threat flags (1 or 0)
    df["has_http"] = urls.str.startswith("http://").astype(int)
    df["has_https"] = urls.str.startswith("https://").astype(int)

    # Check for raw IPv4 address in domain
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    df["is_ip_address"] = domains.str.match(ip_pattern).fillna(False).astype(int)

    # Domain prefix/suffix hyphen check
    df["domain_has_prefix_suffix_hyphen"] = domains.apply(
        lambda x: 1 if x.startswith("-") or x.endswith("-") else 0
    )

    # Known URL shortener flag
    shorteners = r"bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly"
    df["is_shortened"] = domains.str.contains(shorteners).astype(int)

    # Suspicious security keywords
    keywords = r"login|verify|update|account|banking|secure|paypal|signin|confirm"
    df["count_suspicious_keywords"] = urls.str.count(keywords)

    # 5. Shannon entropy
    df["domain_entropy"] = domains.apply(calculate_entropy)

    return df


if __name__ == "__main__":
    sample = pd.DataFrame({"url": [
        "https://www.example.com/index.html",
        "http://192.168.1.1:8080/login?user=a=b",
        "bit.ly/3abcde",
    ]})
    print(extract_url_features(sample, "url").T)
