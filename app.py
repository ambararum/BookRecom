from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi Flask
app = Flask(__name__)

# Load dataset yang sudah diproses
df = pd.read_csv('preprocessed_books.csv')

# Load cosine similarity matrix dan TF-IDF vectorizer
with open('cosine_similarity_matrix.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Fungsi untuk mendapatkan rekomendasi berdasarkan keyword
def get_recommendations_by_keyword(keyword, k=10, threshold=0.3):
    # Preprocessing kata kunci input
    processed_keyword = preprocess_text(keyword)

    # Transform kata kunci menjadi vektor TF-IDF
    keyword_vector = tfidf_vectorizer.transform([processed_keyword])

    # Hitung cosine similarity antara kata kunci dan semua buku
    sim_scores = cosine_similarity(keyword_vector, tfidf_vectorizer.transform(df['content'])).flatten()

    # Filter buku yang similarity-nya melebihi threshold
    recommended_indices = [i for i, score in enumerate(sim_scores) if score >= threshold]

    # Ambil K buku teratas dengan similarity tertinggi
    top_books_indices = sorted(recommended_indices, key=lambda x: sim_scores[x], reverse=True)[:k]

    # Ambil detail buku yang direkomendasikan
    recommended_books = df[['title', 'authors', 'description', 'categories']].iloc[top_books_indices].copy()
    recommended_books['cosine_similarity'] = [sim_scores[i] for i in top_books_indices]

    return recommended_books

# Fungsi preprocessing teks
def preprocess_text(text):
    # Case folding - ubah teks menjadi lowercase
    text = str(text).lower()

    # Tokenization
    words = text.split()

    # Removal punctuation & hanya simpan alfanumerik
    words = [word for word in words if word.isalnum()]

    return ' '.join(words)

# Route utama untuk tampilan pencarian
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    if request.method == "POST":
        keyword = request.form['keyword']
        if keyword:
            recommendations = get_recommendations_by_keyword(keyword)
    
    # Periksa apakah recommendations ada dan tidak kosong
    return render_template("index.html", recommendations=recommendations if recommendations is not None and not recommendations.empty else None)


if __name__ == "__main__":
    app.run(debug=True)
