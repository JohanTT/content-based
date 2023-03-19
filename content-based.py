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
        print("Ma Trận điểm tương đồng\n", self.cosine_sim[1:,1:])

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
            Sử dụng hàm Series của module pandas để lấy ra được vị trí của từng giá trị
            Sử dụng enumerate để thêm giá trị đếm (vị trí) vào bên trong giá trị đang có,
                sau đó biến cặp thành mảng để có thể sắp xếp dễ dàng hơn
            Sử dụng hàm sorted để sắp xếp theo số lớn đến bé:
                + Tham số đầu tiên là danh sách giá trị
                + Tham số thứ hai là một key, key ở đây sử dụng lambda để có thể sắp xếp một đối tượng phức tạp,
                    vì dữ liệu của chúng ta có nhiều cột, nên chúng ta phải chỉ đính danh ra chỉ chọn một cột để tính
                + Tham số thứ ba là chiều sắp xếp, true là sắp xếp giảm dần 
        """
        titles = self.movies['title']
        indices = pd.Series(self.movies.index, index=self.movies['title'])
        idx = indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_x + 1]
        movie_indices = [i[0] for i in sim_scores]
        print (top_x - 1, "tựa phim có điểm tương đồng cao nhất với", title)
        sim_indices = 0
        for indices in movie_indices:
            if titles.iloc[indices] != "title":
                print ("Score:", sim_scores[sim_indices], "Movie:", titles.iloc[indices])
            if sim_indices < len(sim_scores):
                sim_indices += 1

contentBased = CB("movies.csv")
contentBased.refresh()
contentBased.genre_recommendations("Rob Roy (1995)", 6)
