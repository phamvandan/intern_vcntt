from sklearn.feature_extraction.text import CountVectorizer
def count_vec(text):
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    doc_term_matrix= vectorizer.transform(text)
    final=doc_term_matrix.toarray()
    return final

result = count_vec(["The quick brown fox jumped over the lazy dog the.", "no more see the dog."])
print(result)