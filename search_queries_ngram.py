import re
from collections import Counter
import pandas as pd

# read data from the csv
data = pd.read_csv('searchqueries.csv')

# pointing the search queries in the raw data
search_queries = data['Search Query']

print(search_queries)

# defines the ngram 
def generate_ngrams(text, n):
    words = re.findall(r'\w+', text)
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

# gets the top occurred search_queires
def get_top_ngrams(search_queries, n, top_k):
    all_ngrams = []
    for query in search_queries:
        ngrams = generate_ngrams(query.lower(), n)
        all_ngrams.extend(ngrams)

    ngram_counts = Counter(all_ngrams)
    top_ngrams = ngram_counts.most_common(top_k)
    return top_ngrams

# prints the top 10 n keywords which occurres most
top_ngrams = get_top_ngrams(search_queries, n=5, top_k=10)
print(top_ngrams)
