"""
feature_extraction.py
---------------------
Turns raw URL strings into a numeric feature table.

The same functions are used for the training data (data_prep_and_split.py)
and for a single URL typed by a user (predict.py), so the model always sees
features computed in exactly the same way.

Pipeline for every URL:
    raw string -> canonicalise() -> split into host / path / query -> features
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse, ParseResult

import pandas as pd

# ---------------------------------------------------------------------------
# Reference lists used by the threat-specific features
# ---------------------------------------------------------------------------

# Known URL-shortening services. Matched against the *registered domain*
# (e.g. "bit.ly"), never as a substring, so "microsoft.com" is not mistaken
# for "t.co".
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
    "x.co", "s.id", "v.gd", "rebrand.ly", "bl.ink", "lnkd.in", "db.tt",
}

# Top-level domains with a high rate of abuse relative to their size.
# First five: highest-ranked in Interisle's Phishing Landscape 2025
# (Piscitello and Rose 2025). The rest are the formerly free Freenom TLDs and
# other cheap TLDs widely reported as abused when the dataset was collected.
RISKY_TLDS = {
    "xin", "bond", "help", "win", "cfd",
    "tk", "ml", "ga", "cf", "gq",
    "top", "xyz", "icu", "cyou", "sbs", "click", "buzz", "rest", "zip", "mov",
}

# Brands that phishing pages commonly impersonate. Used to detect a brand name
# that appears in a URL without being the site's own registered domain.
BRANDS = {
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook",
    "google", "gmail", "amazon", "facebook", "instagram", "whatsapp",
    "netflix", "ebay", "linkedin", "dropbox", "adobe", "chase",
    "wellsfargo", "bankofamerica", "citibank", "hsbc", "dhl", "fedex",
    "usps", "auspost", "commbank", "westpac", "steam", "coinbase",
}

# Words that create urgency or ask for credentials (social engineering).
SUSPICIOUS_KEYWORDS = r"login|verify|update|account|banking|secure|signin|confirm|password|webscr|ebayisapi"

# Two-label public suffixes, so "example.com.au" gives registered domain
# "example.com.au" rather than "com.au". A short list is enough for the
# features below; it is not a full public-suffix list.
TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "edu.au", "gov.au", "co.nz", "co.jp", "co.in", "com.br", "com.cn",
    "co.za", "com.mx", "com.tr", "com.sg", "com.my", "co.id",
}

SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
WWW_RE = re.compile(r"^www\d?\.", re.IGNORECASE)
IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


# ---------------------------------------------------------------------------
# Step 1: canonicalise and parse
# ---------------------------------------------------------------------------

def canonicalise(url):
    """
    Bring every URL to one consistent format before any feature is computed.

    - removes the scheme (http://, https://, ftp://) because in this dataset
      its presence depends on which source list a URL came from, not on
      whether the URL is malicious (see the report, Section 5.2)
    - removes a leading "www." for the same reason
    - lower-cases the host (DNS is case-insensitive); the path keeps its case
    """
    s = str(url).strip()
    s = SCHEME_RE.sub("", s)
    s = WWW_RE.sub("", s)
    # Lower-case only the host part (everything before the first / ? or #)
    m = re.match(r"^([^/?#]*)(.*)$", s, re.DOTALL)
    host, rest = m.group(1), m.group(2)
    return host.lower() + rest


def safe_url_parse(canonical_url):
    """
    Parse a canonical (scheme-less) URL without crashing on malformed input.

    urlparse() raises ValueError on unbalanced square brackets, which occur in
    a few raw rows. Those brackets are percent-encoded and parsing is retried;
    a string that still fails is treated as a bare path.
    """
    s = "//" + canonical_url  # "//" makes urlparse treat the first part as the host
    try:
        return urlparse(s)
    except ValueError:
        try:
            return urlparse(s.replace("[", "%5B").replace("]", "%5D"))
        except ValueError:
            return ParseResult("", "", canonical_url, "", "", "")


def split_host(host):
    """
    Split a host name into (subdomain, registered_domain, tld).

    "login.paypal.com.evil.xyz" -> ("login.paypal.com", "evil.xyz", "xyz")
    "shop.example.com.au"       -> ("shop", "example.com.au", "au")
    An IP address is returned as its own registered domain with no TLD.
    """
    host = host.split("@")[-1]          # drop any "user@" part
    host = host.split(":")[0].strip(".")  # drop any ":port"
    if not host or IPV4_RE.match(host):
        return "", host, ""
    labels = host.split(".")
    if len(labels) == 1:
        return "", host, ""
    n = 3 if ".".join(labels[-2:]) in TWO_LEVEL_SUFFIXES and len(labels) >= 3 else 2
    return ".".join(labels[:-n]), ".".join(labels[-n:]), labels[-1]


def shannon_entropy(text):
    """Shannon entropy in bits per character: high for random strings."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


# ---------------------------------------------------------------------------
# Step 2: features for one URL
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "url_len", "domain_len", "path_len",
    "count_dots", "count_hyphens", "count_slashes", "count_questions",
    "count_equal", "count_at", "count_digits", "domain_count_dots",
    "is_ip_address", "domain_has_prefix_suffix_hyphen", "is_shortened",
    "count_suspicious_keywords", "domain_entropy",
    "risky_tld", "has_punycode_or_non_ascii", "brand_outside_domain",
]


def url_features(raw_url):
    """Return a dict of every feature in FEATURE_COLUMNS for one raw URL."""
    canon = canonicalise(raw_url)
    parsed = safe_url_parse(canon)
    host = parsed.netloc.split("@")[-1].split(":")[0]
    subdomain, registered, tld = split_host(parsed.netloc)
    lower = canon.lower()

    # Brand impersonation: a known brand appears in the subdomain or the path,
    # but the site's registered name (e.g. "evil" in "evil.xyz") is not that
    # brand. "paypal.com.evil.xyz/login" -> 1; "paypal.com/login" -> 0.
    registered_name = registered.split(".")[0] if registered else ""
    outside_text = (subdomain + "/" + parsed.path + "?" + parsed.query).lower()
    brand_outside = int(any(b in outside_text and b != registered_name for b in BRANDS))

    return {
        # Lengths: long URLs push the real domain out of view
        "url_len": len(canon),
        "domain_len": len(host),
        "path_len": len(parsed.path),
        # Character counts
        "count_dots": canon.count("."),
        "count_hyphens": canon.count("-"),
        "count_slashes": canon.count("/"),
        "count_questions": canon.count("?"),
        "count_equal": canon.count("="),
        "count_at": canon.count("@"),       # browser ignores text before @
        "count_digits": sum(ch.isdigit() for ch in canon),
        "domain_count_dots": host.count("."),  # subdomain depth
        # Structural threat flags
        "is_ip_address": int(bool(IPV4_RE.match(host))),
        "domain_has_prefix_suffix_hyphen": int(host.startswith("-") or host.endswith("-")),
        "is_shortened": int(registered in SHORTENERS),
        "count_suspicious_keywords": len(re.findall(SUSPICIOUS_KEYWORDS, lower)),
        "domain_entropy": round(shannon_entropy(host), 4),
        # Researched features
        "risky_tld": int(tld in RISKY_TLDS),
        "has_punycode_or_non_ascii": int("xn--" in host or any(ord(ch) > 127 for ch in host)),
        "brand_outside_domain": brand_outside,
    }


# ---------------------------------------------------------------------------
# Step 3: features for a whole DataFrame
# ---------------------------------------------------------------------------

def extract_url_features(df, url_column="url"):
    """
    Add every feature column to a copy of df and return it.

    The original columns (e.g. url, label, type) are kept so they can be saved
    alongside the features; the training scripts select FEATURE_COLUMNS only.
    """
    feats = pd.DataFrame([url_features(u) for u in df[url_column].astype(str)],
                         index=df.index, columns=FEATURE_COLUMNS)
    return pd.concat([df.copy(), feats], axis=1)


if __name__ == "__main__":
    # Quick demonstration on a few hand-written URLs
    demo = pd.DataFrame({"url": [
        "https://www.google.com",
        "google.com",
        "http://paypal.com.secure-login.xyz/verify?id=123",
        "192.168.1.1/admin/login.php",
        "bit.ly/3xYz89",
        "http://xn--pple-43d.com/signin",
    ]})
    pd.set_option("display.width", 200)
    print(extract_url_features(demo)[["url", "url_len", "is_ip_address", "is_shortened",
                                      "risky_tld", "has_punycode_or_non_ascii",
                                      "brand_outside_domain"]])
