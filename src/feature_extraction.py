import math
from collections import Counter
from urllib.parse import urlparse, ParseResult
import numpy as np
import pandas as pd

def safe_url_parse(url):
    """
    Safely parses URLs, handling malformed IPv6 brackets or noisy characters
    commonly found in raw cybersecurity datasets.
    """
    url_str = str(url).strip()
    
    # Ensure netloc parsing works for schemeless URLs
    if '://' not in url_str:
        url_str = '//' + url_str

    try:
        return urlparse(url_str)
    except ValueError:
        # Sanitize malformed square brackets (convert [ and ] to percent-encoding)
        sanitized = url_str.replace('[', '%5B').replace(']', '%5D')
        try:
            return urlparse(sanitized)
        except Exception:
            # Return empty fallback ParseResult if string is completely unparseable
            return ParseResult(scheme='', netloc='', path=url_str, params='', query='', fragment='')

def calculate_entropy(text):
    """Calculates the Shannon entropy of a string (higher means more random)."""
    if not text:
        return 0.0
    counts = Counter(text)
    frequencies = [float(c) / len(text) for c in counts.values()]
    return -sum(f * math.log2(f) for f in frequencies)

def extract_url_features(df, url_column):
    """
    Transforms a DataFrame column of raw URLs into numerical feature columns.
    """
    urls = df[url_column].astype(str)

    # 1. Safe Parse
    parsed_urls = urls.apply(safe_url_parse)

    domains = parsed_urls.apply(lambda x: x.netloc)
    paths = parsed_urls.apply(lambda x: x.path)

    # 2. Lexical Length Features
    df['url_len'] = urls.str.len()
    df['domain_len'] = domains.str.len()
    df['path_len'] = paths.str.len()

    # 3. Special Character Counts
    df['count_dots'] = urls.str.count(r'\.')
    df['count_hyphens'] = urls.str.count('-')
    df['count_slashes'] = urls.str.count('/')
    df['count_questions'] = urls.str.count(r'\?')
    df['count_equal'] = urls.str.count('=')
    df['count_at'] = urls.str.count('@')
    df['count_digits'] = urls.str.count(r'\d')

    # Subdomain depth
    df['domain_count_dots'] = domains.str.count(r'\.')

    # 4. Structural & Security Threat Flags
    df['has_http'] = urls.str.lower().str.startswith('http://').astype(int)
    df['has_https'] = urls.str.lower().str.startswith('https://').astype(int)

    # Check for raw IPv4 address in domain
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    df['is_ip_address'] = domains.str.match(ip_pattern).fillna(0).astype(int)

    # Domain prefix/suffix hyphen check
    df['domain_has_prefix_suffix_hyphen'] = domains.apply(
        lambda x: 1 if x.startswith('-') or x.endswith('-') else 0
    )

    # Known URL shortener flag
    shorteners = r'bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly'
    df['is_shortened'] = domains.str.lower().str.contains(shorteners).astype(int)

    # Suspicious security keywords
    keywords = r'login|verify|update|account|banking|secure|paypal|signin|confirm'
    df['count_suspicious_keywords'] = urls.str.lower().str.count(keywords)

    # 5. Shannon Entropy
    df['domain_entropy'] = domains.apply(calculate_entropy)

    return df

if __name__ == '__main__':
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'datasets/raw_urls.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'datasets/processed_features.csv'

    print(f"Reading raw data from {input_file}...")
    try:
        raw_data = pd.read_csv(input_file)
        print("Extracting features...")
        processed_data = extract_url_features(raw_data, url_column='url')
        processed_data.to_csv(output_file, index=False)
        print(f"Successfully saved extracted feature matrix to {output_file}")
    except FileNotFoundError:
        print(f"File '{input_file}' not found. Executing sample feature extraction test...")
        sample_df = pd.DataFrame({
            'url': [
                'https://google.com',
                'http://[malicious-domain].com/login',
                '192.168.1.1/login/verify.php',
                'https://bit.ly/3xYz89'
            ]
        })
        res = extract_url_features(sample_df, url_column='url')
        print("\nExtracted Features Preview:")
        print(res[['url', 'url_len', 'is_ip_address', 'is_shortened', 'domain_entropy']])
