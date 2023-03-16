import pandas as pd
from matrix import tfidf_matrix, cosine_sim
from readcsv import get_dataframe_movies_csv

class CB(object):
    """
        Khởi tại dataframe "movies" với hàm "get_dataframe_movies_csv"
    """
    def __init__(self, movies_csv):
        self.movies = get_dataframe_movies_csv(movies_csv)
        self.tfidf_matrix = None
        self.cosine_sim = None

    def build_model(self):
        """
            Tách các giá trị của genres ở từng bộ phim đang được ngăn cách bởi '|'
        """
        self.movies['genres'] = self.movies['genres'].str.split('|')
        self.movies['genres'] = self.movies['genres'].fillna("").astype('str')
        self.tfidf_matrix = tfidf_matrix(self.movies)
        #print("Ma trận TF-IDF\n", self.tfidf_matrix)
        self.cosine_sim = cosine_sim(self.tfidf_matrix)
        print("Ma Trận chuẩn hoá\n", self.cosine_sim)

    def refresh(self):
        """
             Chuẩn hóa dữ liệu và tính toán lại ma trận
        """
        self.build_model()

    def fit(self):
        self.refresh()
        
    def genre_recommendations(self, title, top_x):
        """
            Xây dựng hàm trả về danh sách top film tương đồng theo tên film truyền vào:
            + Tham số truyền vào gồm "title" là tên film và "topX" là top film tương đồng cần lấy
            + Tạo ra list "sim_score" là danh sách điểm tương đồng với film truyền vào
            + Sắp xếp điểm tương đồng từ cao đến thấp
            + Trả về top danh sách tương đồng cao nhất theo giá trị "topX" truyền vào
        """
        titles = self.movies['title']
        indices = pd.Series(self.movies.index, index=self.movies['title'])
        idx = indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_x + 1]
        movie_indices = [i[0] for i in sim_scores]
        sim_indices = 0
        for indices in movie_indices:
            if titles.iloc[indices] != "title":
                print ("Score:", sim_scores[sim_indices], "Movie:", titles.iloc[indices])
            if sim_indices < len(sim_scores):
                sim_indices += 1

contentBased = CB("movies.csv")
contentBased.refresh()
contentBased.genre_recommendations("Rob Roy (1995)", 6)
