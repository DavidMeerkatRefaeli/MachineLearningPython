# Spam Classification
with open('./Data/emailSample1.txt', 'r') as file:
    file_contents = file.read()


def get_vocab_list():
    pass


def process_email(file_contents):
    vocab_list = get_vocab_list()
    word_indices = []


word_indices = process_email(file_contents)