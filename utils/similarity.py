from utils.text_processor import TextProcessor

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def calculate_similarity_matrix(texts, n=3):
    """Compute Jaccard similarity matrix using n-grams"""
    text_processor = TextProcessor()
    ngram_sets = [set(text_processor.extract_ngrams(text, n)) for text in texts]

    similarity_matrix = []
    for i in range(len(ngram_sets)):
        row = []
        for j in range(len(ngram_sets)):
            if i == j:
                row.append(100.0)
            else:
                sim = jaccard_similarity(ngram_sets[i], ngram_sets[j])
                row.append(round(sim * 100, 2))
        similarity_matrix.append(row)
    
    return similarity_matrix