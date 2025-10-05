#src/preprocess.py
import pandas as pd
import re
import argparse
from urllib.parse import urlparse
import math
from collections import Counter

# Suspicious keywords often used in phishing URLs
KEYWORDS = ['login', 'secure', 'update', 'verify', 'bank', 'account', 'signin', 'password']

def shannon_entropy(string):
    """Calculate Shannon entropy of a string."""
    if not string:
        return 0
    probs = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum([p * math.log(p, 2) for p in probs])

def has_ip_address(url):
    """Check if URL contains an IP address."""
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    return 1 if re.search(ip_pattern, url) else 0

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc or url.split('/')[0]

    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['has_ip'] = has_ip_address(url)
    features['has_at'] = 1 if '@' in url else 0
    features['prefix_suffix'] = 1 if '-' in domain else 0
    features['subdomain_count'] = domain.count('.') - 1
    features['https_in_domain'] = 1 if 'https' in domain else 0
    features['entropy'] = shannon_entropy(domain)

    # Keyword features
    for kw in KEYWORDS:
        features[f'kw_{kw}'] = 1 if kw in url.lower() else 0

    return features

def main(args):
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} URLs")

    # Convert labels: 'bad' → 1, 'good' → 0
    df['Label'] = df['Label'].map({'bad': 1, 'good': 0})

    # Drop missing values & duplicates
    df = df.dropna().drop_duplicates()

    # Optional sampling to make processing faster
    if args.sample_size:
        df = df.sample(n=int(args.sample_size), random_state=42)

    features = df['URL'].apply(extract_features).apply(pd.Series)
    features['label'] = df['Label']

    features.to_csv(args.output, index=False)
    print(f"Saved processed features to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to raw dataset CSV')
    p.add_argument('--output', required=True, help='Path to save processed features')
    p.add_argument('--sample_size', required=False, help='Optional: limit sample size')
    args = p.parse_args()
    main(args)
