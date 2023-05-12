from pandas import read_csv
def get_dataframe_movies_csv(text):
    """
    Sử dụng hàm read_csv của pandas để đọc file csv của movilens, 
        lưu thành dataframe với 3 cột user id, title, genres
    Tham số đầu tiên là file csv
    Tham số thứ hai là cách cột cách nhau bởi dấu gì
    Tham số thứ ba là khi đọc file csv xong nó sẽ đưa ra cột tên gì
    Tham số cuối là dạng mã hóa ký tự
    """
    movie_cols = ['movie_id', 'title', 'genres','directedBy','starring']
    movies = read_csv(text, sep=',', names=movie_cols, encoding='latin-1')
    return movies