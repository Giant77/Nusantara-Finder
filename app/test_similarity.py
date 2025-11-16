"""
Unit testing untuk memvalidasi perhitungan similarity
Memastikan semua metode menghasilkan skor antara 0-1 (0-100%)
"""

import unittest
from collections import defaultdict
from similarity_calculator import SimilarityCalculator


class TestSimilarityCalculator(unittest.TestCase):
    """Test case untuk SimilarityCalculator"""
    
    def setUp(self):
        """Setup test data"""
        # Sample inverted index
        self.inverted_index = {
            1: {'wisata': 3, 'pantai': 2, 'bali': 1, 'indah': 1},
            2: {'wisata': 2, 'gunung': 3, 'bromo': 2, 'indah': 1},
            3: {'kuliner': 2, 'makanan': 3, 'indonesia': 1, 'enak': 2},
            4: {'pantai': 2, 'bali': 3, 'sunset': 1, 'indah': 2},
            5: {'museum': 2, 'sejarah': 3, 'jakarta': 1, 'nasional': 1}
        }
        
        # Sample combined data
        self.combined_data = {
            1: {'category': 'Wisata', 'title': 'Pantai Bali', 'date': '2024-01-01',
                'image_url': '', 'url': '', 'content': 'Pantai indah di Bali'},
            2: {'category': 'Wisata', 'title': 'Gunung Bromo', 'date': '2024-01-02',
                'image_url': '', 'url': '', 'content': 'Gunung yang indah'},
            3: {'category': 'Kuliner', 'title': 'Makanan Indonesia', 'date': '2024-01-03',
                'image_url': '', 'url': '', 'content': 'Kuliner enak'},
            4: {'category': 'Wisata', 'title': 'Pantai Bali Sunset', 'date': '2024-01-04',
                'image_url': '', 'url': '', 'content': 'Sunset di Bali'},
            5: {'category': 'Sejarah', 'title': 'Museum Nasional', 'date': '2024-01-05',
                'image_url': '', 'url': '', 'content': 'Museum sejarah'}
        }
        
        self.calculator = SimilarityCalculator(self.inverted_index, self.combined_data)
    
    def test_jaccard_perfect_match(self):
        """Test Jaccard dengan query yang sama persis dengan dokumen"""
        query_words = ['wisata', 'pantai', 'bali', 'indah']
        doc_words_freq = self.inverted_index[1]
        
        similarity = self.calculator.calculate_jaccard(query_words, doc_words_freq)
        
        # Perfect match harus menghasilkan similarity 1.0
        self.assertEqual(similarity, 1.0)
    
    def test_jaccard_no_match(self):
        """Test Jaccard dengan query yang tidak ada kesamaan"""
        query_words = ['museum', 'sejarah']
        doc_words_freq = self.inverted_index[1]  # Dokumen tentang pantai
        
        similarity = self.calculator.calculate_jaccard(query_words, doc_words_freq)
        
        # Tidak ada match harus menghasilkan similarity 0.0
        self.assertEqual(similarity, 0.0)
    
    def test_jaccard_partial_match(self):
        """Test Jaccard dengan partial match"""
        query_words = ['pantai', 'indah']
        doc_words_freq = self.inverted_index[1]
        
        similarity = self.calculator.calculate_jaccard(query_words, doc_words_freq)
        
        # Harus di antara 0 dan 1
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_cosine_perfect_match(self):
        """Test Cosine dengan query yang identik"""
        query_words = ['wisata', 'wisata', 'wisata', 'pantai', 'pantai', 'bali', 'indah']
        doc_words_freq = self.inverted_index[1]
        
        similarity = self.calculator.calculate_cosine(query_words, doc_words_freq)
        
        # Perfect match harus menghasilkan similarity 1.0
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_cosine_no_match(self):
        """Test Cosine dengan query yang tidak ada kesamaan"""
        query_words = ['museum', 'sejarah']
        doc_words_freq = self.inverted_index[1]
        
        similarity = self.calculator.calculate_cosine(query_words, doc_words_freq)
        
        # Tidak ada match harus menghasilkan similarity 0.0
        self.assertEqual(similarity, 0.0)
    
    def test_cosine_partial_match(self):
        """Test Cosine dengan partial match"""
        query_words = ['pantai', 'indah']
        doc_words_freq = self.inverted_index[1]
        
        similarity = self.calculator.calculate_cosine(query_words, doc_words_freq)
        
        # Harus di antara 0 dan 1
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_bm25_score_range(self):
        """Test BM25 menghasilkan skor dalam range 0-1"""
        query_words = ['pantai', 'bali']
        
        bm25_scores = self.calculator.calculate_bm25(query_words)
        
        # Semua skor harus antara 0 dan 1
        for doc_id, score in bm25_scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_search_returns_valid_results(self):
        """Test pencarian menghasilkan hasil yang valid"""
        results = self.calculator.search('pantai bali', algorithm='jaccard')
        
        # Harus ada hasil
        self.assertGreater(len(results), 0)
        
        # Setiap hasil harus punya similarity antara 0-1
        for result in results:
            self.assertIn('similarity', result)
            self.assertGreaterEqual(result['similarity'], 0.0)
            self.assertLessEqual(result['similarity'], 1.0)
    
    def test_search_sorted_by_similarity(self):
        """Test hasil pencarian diurutkan berdasarkan similarity"""
        results = self.calculator.search('pantai indah', algorithm='cosine')
        
        # Hasil harus diurutkan descending
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]['similarity'], results[i+1]['similarity'])
    
    def test_search_with_category_filter(self):
        """Test pencarian dengan filter kategori"""
        results = self.calculator.search('wisata', algorithm='jaccard', target_category='Wisata')
        
        # Semua hasil harus kategori Wisata
        for result in results:
            self.assertEqual(result['category'], 'Wisata')
    
    def test_all_algorithms_produce_valid_scores(self):
        """Test semua algoritma menghasilkan skor valid"""
        query = 'pantai bali indah'
        algorithms = ['jaccard', 'cosine', 'bm25']
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                results = self.calculator.search(query, algorithm=algorithm)
                
                # Semua hasil harus punya skor 0-1
                for result in results:
                    self.assertGreaterEqual(result['similarity'], 0.0)
                    self.assertLessEqual(result['similarity'], 1.0)
    
    def test_empty_query(self):
        """Test dengan query kosong"""
        results = self.calculator.search('', algorithm='jaccard')
        
        # Query kosong harus return list kosong
        self.assertEqual(len(results), 0)
    
    def test_min_similarity_filter(self):
        """Test filter minimum similarity"""
        results = self.calculator.search('pantai', algorithm='jaccard', min_similarity=0.5)
        
        # Semua hasil harus punya similarity >= 0.5
        for result in results:
            self.assertGreaterEqual(result['similarity'], 0.5)


def run_comparison_test():
    """
    Menjalankan test perbandingan ketiga metode
    Menampilkan skor untuk query yang sama
    """
    print("\n" + "="*60)
    print("PERBANDINGAN METODE SIMILARITY")
    print("="*60)
    
    # Setup test data
    inverted_index = {
        1: {'wisata': 3, 'pantai': 2, 'bali': 1, 'indah': 1, 'matahari': 1, 'terbenam': 1},
        2: {'wisata': 2, 'gunung': 3, 'bromo': 2, 'indah': 1, 'pemandangan': 2},
        3: {'kuliner': 2, 'makanan': 3, 'indonesia': 1, 'enak': 2, 'tradisional': 1},
        4: {'pantai': 3, 'bali': 2, 'sunset': 1, 'indah': 2, 'romantis': 1},
        5: {'museum': 2, 'sejarah': 3, 'jakarta': 1, 'nasional': 1, 'budaya': 2}
    }
    
    combined_data = {
        1: {'category': 'Wisata', 'title': 'Pantai Bali Sunset'},
        2: {'category': 'Wisata', 'title': 'Gunung Bromo'},
        3: {'category': 'Kuliner', 'title': 'Makanan Indonesia'},
        4: {'category': 'Wisata', 'title': 'Pantai Kuta Bali'},
        5: {'category': 'Sejarah', 'title': 'Museum Nasional'}
    }
    
    calculator = SimilarityCalculator(inverted_index, combined_data)
    
    test_queries = [
        'pantai bali indah',
        'wisata gunung',
        'kuliner indonesia',
        'museum sejarah'
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        for algorithm in ['jaccard', 'cosine', 'bm25']:
            results = calculator.search(query, algorithm=algorithm)
            print(f"\n{algorithm.upper()}:")
            
            for i, result in enumerate(results[:3], 1):
                similarity_pct = result['similarity'] * 100
                print(f"  {i}. {result['title']}: {similarity_pct:.2f}%")


if __name__ == '__main__':
    # Jalankan unit tests
    print("Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Jalankan comparison test
    run_comparison_test()