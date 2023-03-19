from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_matrix(movies):
    """
        Dùng hàm "TfidfVectorizer" để chuẩn hóa "genres" với:
        + analyzer='word': chọn đơn vị trích xuất là word
        + ngram_range=(1, 1): mỗi lần trích xuất 1 word
        + min_df=0: tỉ lệ word không đọc được là 0
        Lúc này ma trận trả về với số dòng tương ứng với số lượng film 
            và số cột tương ứng với số từ được tách ra từ "genres"
        Phương thức fit_transform là sự kết hợp giữa phương thức fit() và transform():
        + Phương thức fit() giúp khớp dữ liệu vào một mô hình
        + Phương thức transform() giúp chuyển đổi dữ liệu thành một dạng phù hợp hơn với mô hình
    """
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0)
    new_tfidf_matrix = tf.fit_transform(movies['genres']).toarray()
    print("Ma trận TF-IDF\n", new_tfidf_matrix[1:,1:])
    return new_tfidf_matrix

from sklearn.metrics.pairwise import linear_kernel
def cosine_sim(matrix):
    """
            Dùng hàm "linear_kernel" để tạo thành ma trận 
                hình vuông với số hàng và số cột là số lượng film
                để tính toán điểm tương đồng giữa từng bộ phim với nhau
            Bên trong hàm sẽ truyền vào 2 tham số:
            + Tham số đầu tiên là một ma trận tính năng
            + Tham số thứ hai là một ma trận tính năng thứ 2 tùy chọn
    """
    new_cosine_sim = linear_kernel(matrix, matrix)
    return new_cosine_sim