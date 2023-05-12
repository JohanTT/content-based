from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
from content_based import contentBased

app = Flask(__name__)

# Đọc dữ liệu từ file csv
movies_df = pd.read_csv('csv_data/movies.csv')

@app.route('/')
def hello_world():
    # Trích xuất thông tin các cột cần thiết và chuyển đổi thành list các dict
    movies = []
    for idx, row in movies_df.iterrows():
        movie = {
            'id': row['movieId'],
            'title': row['title'],
            'genres': row['genres'],
            'directedBy': row['directedBy'],
            'starring': row['starring']
            # Thêm các thuộc tính khác nếu cần
        }
        movies.append(movie)
    
    return render_template('movies.html', movies=movies)

@app.route('/<title>', methods=['POST'])
def recommended(title):
    category = request.form.get('category')
    recommend = contentBased.CB('csv_data/movies.csv')
    recommend.refresh(category)
    result = recommend.genre_recommendations(title, 10)
    print(category)
    return render_template('recommend.html', movies=result)

if __name__ == '__main__':
    app.run()
