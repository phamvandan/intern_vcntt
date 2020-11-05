# Hashing Vectorization
from sklearn.feature_extraction.text import HashingVectorizer

def hash_vec(text):
    vectorizer = HashingVectorizer(n_features=10)
    doc_term_matrix= vectorizer.transform(text)
    final=doc_term_matrix.toarray()
    return final

result = hash_vec(["The car is driven on the road", "The truck is driven on the highway"])
print(result.shape)
print(result)