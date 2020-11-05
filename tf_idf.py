
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_vec(text):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    doc_term_matrix= vectorizer.transform(text)
    final=doc_term_matrix.toarray()
    return final

corpus =  ["the car is driven on the road", 'the truck is driven on the highway']


result = tf_idf_vec(corpus)

print(result)