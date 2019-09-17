def load_movie_list():
    with open ("./Data/movie_ids.txt", "r") as myfile:
        names = [line.split(' ', 1)[1].strip() for line in myfile.readlines()]
    return names