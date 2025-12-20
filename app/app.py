from flask import Flask, render_template, request
import csv
import os
from collections import defaultdict
import re
from similarity_calculator import SimilarityCalculator

app = Flask(__name__)

# Tentukan path file berdasarkan lokasi app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, '..', 'inverted_index.txt')
COMBINED_DATA_PATH = os.path.join(BASE_DIR, '..', 'merged_combined_data.csv')


def load_inverted_index(file_path):
    """
    Memuat inverted index dari file
    Format: word: (doc_id, freq), (doc_id, freq), ...
    
    Returns:
        dict: {doc_id: {word: frequency}}
    """
    inverted_index = defaultdict(lambda: defaultdict(int))
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(': ')
            if len(parts) != 2:
                continue
            
            word, postings = parts
            doc_list = re.findall(r'\((\d+), (\d+)\)', postings)
            
            for doc_id, freq in doc_list:
                inverted_index[int(doc_id)][word] = int(freq)
    
    return dict(inverted_index)


def load_combined_data(file_path):
    """
    Memuat data dokumen dari CSV
    
    Returns:
        dict: {doc_id: {field: value}}
    """
    combined_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            doc_id = int(row['Doc ID'])
            combined_data[doc_id] = {
                'category': row['Category'],
                'title': row['Title_clean'],
                'date': row['Date'],
                'image_url': row['Image URL'],
                'url': row['URL'],
                'content': row['Content'],
            }
    
    return combined_data


# Muat data saat aplikasi dimulai
print("Loading inverted index...")
inverted_index = load_inverted_index(INDEX_PATH)
print(f"Loaded {len(inverted_index)} documents")

print("Loading combined data...")
combined_data = load_combined_data(COMBINED_DATA_PATH)
print(f"Loaded {len(combined_data)} document details")

# Inisialisasi Similarity Calculator
print("Initializing similarity calculator...")
similarity_calc = SimilarityCalculator(inverted_index, combined_data)
print("Ready!")


@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search_results():
    """
    Halaman hasil pencarian dengan pagination
    """
    # Ambil parameter pencarian - support both POST and GET
    if request.method == 'POST':
        query = request.form.get('query')
        category = request.form.get('category', None)
        algorithm = request.form.get('algorithm', 'jaccard')
    else:  # GET request (dari pagination)
        query = request.args.get('query')
        category = request.args.get('category', None)
        algorithm = request.args.get('algorithm', 'jaccard')
    
    # Validasi algorithm
    if algorithm not in ['jaccard', 'cosine', 'bm25']:
        algorithm = 'jaccard'
    
    # Lakukan pencarian
    all_results = similarity_calc.search(
        query=query,
        algorithm=algorithm,
        target_category=category,
        min_similarity=0.01  # Filter hasil dengan similarity minimal 1%
    )
    
    # Pagination setup
    results_per_page = 10
    page = request.args.get('page', 1, type=int)
    total_results = len(all_results)
    total_pages = (total_results + results_per_page - 1) // results_per_page  # Ceiling division
    
    # Validasi halaman
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    # Ambil hasil untuk halaman saat ini
    start = (page - 1) * results_per_page
    end = start + results_per_page
    paginated_results = all_results[start:end]
    
    # Generate page range untuk pagination (maksimal 5 angka)
    page_range = range(max(1, page - 2), min(total_pages, page + 2) + 1)
    
    return render_template(
        'result.html',
        query=query,
        results=paginated_results,
        total_results=total_results,
        total_pages=total_pages,
        current_page=page,
        page_range=page_range,
        category=category,
        algorithm=algorithm
    )


@app.route('/content/<int:content_id>')
def content(content_id):
    """
    Halaman detail konten
    """
    doc = combined_data.get(content_id, {})
    
    if not doc:
        return render_template('404.html'), 404
    
    return render_template(
        'content.html',
        content=doc,
        query=request.args.get('query', '')
    )


@app.errorhandler(404)
def page_not_found(e):
    """Handler untuk halaman tidak ditemukan"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Handler untuk internal server error"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)