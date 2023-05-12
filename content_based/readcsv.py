from pandas import read_csv
def get_dataframe_movies_csv(text):
    """
    đọc file csv của movilens, lưu thành dataframe với 3 cột user id, title, genres
    """
    movie_cols = ['movie_id', 'title', 'genres','directedBy','starring']
    movies = read_csv(text, sep=',', names=movie_cols, encoding='latin-1')
    return movies