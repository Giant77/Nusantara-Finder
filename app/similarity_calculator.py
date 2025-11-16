"""
Modul untuk perhitungan similarity dengan normalisasi 0-100%
Mendukung Jaccard, Cosine Similarity, dan BM25
"""

import math
from collections import Counter
from rank_bm25 import BM25Okapi


class SimilarityCalculator:
    """Class untuk menghitung berbagai jenis similarity"""
    
    def __init__(self, inverted_index, combined_data):
        self.inverted_index = inverted_index
        self.combined_data = combined_data
        self.bm25_model = None
        self.bm25_id_map = []
        self._build_bm25()
    
    def _build_bm25(self):
        """Membangun model BM25 dari inverted index"""
        bm25_corpus = []
        
        for doc_id, freq_map in self.inverted_index.items():
            tokens = []
            for word, freq in freq_map.items():
                tokens.extend([word] * freq)
            bm25_corpus.append(tokens)
            self.bm25_id_map.append(doc_id)
        
        self.bm25_model = BM25Okapi(bm25_corpus)
    
    def preprocess_query(self, query):
        """Preprocessing query menjadi list of words"""
        # Lowercase dan split
        words = query.lower().split()
        # Filter kata yang terlalu pendek (opsional)
        words = [w for w in words if len(w) > 1]
        return words
    
    def calculate_jaccard(self, query_words, doc_words_freq):
        """
        Menghitung Jaccard Similarity
        Jaccard = |A ∩ B| / |A ∪ B|
        
        Args:
            query_words: List kata dari query
            doc_words_freq: Dictionary {word: frequency} dari dokumen
        
        Returns:
            float: Similarity score antara 0-1
        """
        query_set = set(query_words)
        doc_set = set(doc_words_freq.keys())
        
        # Hitung intersection dan union
        intersection = len(query_set & doc_set)
        union = len(query_set | doc_set)
        
        # Hindari pembagian dengan nol
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_cosine(self, query_words, doc_words_freq):
        """
        Menghitung Cosine Similarity dengan TF (Term Frequency)
        Cosine = (A · B) / (||A|| × ||B||)
        
        Args:
            query_words: List kata dari query
            doc_words_freq: Dictionary {word: frequency} dari dokumen
        
        Returns:
            float: Similarity score antara 0-1
        """
        # Hitung frekuensi kata dalam query
        query_freq = Counter(query_words)
        
        # Hitung dot product (A · B)
        dot_product = 0
        for word in query_freq:
            if word in doc_words_freq:
                dot_product += query_freq[word] * doc_words_freq[word]
        
        # Hitung magnitude ||A|| untuk query
        query_magnitude = math.sqrt(sum(freq ** 2 for freq in query_freq.values()))
        
        # Hitung magnitude ||B|| untuk dokumen
        doc_magnitude = math.sqrt(sum(freq ** 2 for freq in doc_words_freq.values()))
        
        # Hindari pembagian dengan nol
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def calculate_bm25(self, query_words):
        """
        Menghitung BM25 score untuk semua dokumen
        BM25 menggunakan implementasi dari rank_bm25
        
        Args:
            query_words: List kata dari query
        
        Returns:
            dict: {doc_id: normalized_score} antara 0-1
        """
        # Dapatkan raw scores dari BM25
        raw_scores = self.bm25_model.get_scores(query_words)
        
        # Normalisasi scores ke range 0-1
        max_score = max(raw_scores) if len(raw_scores) > 0 else 1
        min_score = min(raw_scores) if len(raw_scores) > 0 else 0
        
        # Hindari pembagian dengan nol
        score_range = max_score - min_score
        if score_range == 0:
            # Jika semua score sama, berikan nilai 0 untuk semua
            normalized_scores = {self.bm25_id_map[i]: 0.0 for i in range(len(raw_scores))}
        else:
            # Normalisasi min-max ke range 0-1
            normalized_scores = {
                self.bm25_id_map[i]: (raw_scores[i] - min_score) / score_range
                for i in range(len(raw_scores))
            }
        
        return normalized_scores
    
    def search(self, query, algorithm='jaccard', target_category=None, min_similarity=0.0):
        """
        Melakukan pencarian dengan algoritma yang dipilih
        
        Args:
            query: String query pencarian
            algorithm: 'jaccard', 'cosine', atau 'bm25'
            target_category: Filter kategori (opsional)
            min_similarity: Threshold minimum similarity (default 0.0)
        
        Returns:
            list: List of dict hasil pencarian dengan similarity 0-100%
        """
        # Preprocess query
        query_words = self.preprocess_query(query)
        
        if not query_words:
            return []
        
        results = []
        
        # Untuk BM25, hitung semua scores sekaligus
        if algorithm == 'bm25':
            bm25_scores = self.calculate_bm25(query_words)
        
        # Iterasi setiap dokumen
        for doc_id, doc_words_freq in self.inverted_index.items():
            doc_data = self.combined_data.get(doc_id, {})
            
            # Filter berdasarkan kategori
            if target_category:
                doc_category = doc_data.get('category', '').lower()
                if doc_category != target_category.lower():
                    continue
            
            # Hitung similarity berdasarkan algoritma
            if algorithm == 'jaccard':
                similarity = self.calculate_jaccard(query_words, doc_words_freq)
            elif algorithm == 'cosine':
                similarity = self.calculate_cosine(query_words, doc_words_freq)
            elif algorithm == 'bm25':
                similarity = bm25_scores.get(doc_id, 0.0)
            else:
                similarity = 0.0
            
            # Filter berdasarkan minimum similarity
            if similarity >= min_similarity:
                results.append({
                    'content_id': doc_id,
                    'title': doc_data.get('title', 'Unknown Title'),
                    'similarity': similarity,  # Nilai 0-1
                    'category': doc_data.get('category', 'Unknown Category'),
                    'date': doc_data.get('date', 'Unknown Date'),
                    'image_url': doc_data.get('image_url', ''),
                    'url': doc_data.get('url', ''),
                    'content': doc_data.get('content', '')
                })
        
        # Sort berdasarkan similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results


def format_similarity_percentage(similarity_score):
    """
    Format similarity score (0-1) menjadi persentase string
    
    Args:
        similarity_score: float antara 0-1
    
    Returns:
        str: Persentase dengan 2 desimal (contoh: "85.50%")
    """
    return f"{similarity_score * 100:.2f}%"