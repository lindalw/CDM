from Vocab import Vocab


vocab = Vocab(file = 'vocab.csv')

# a = vocab.word2index
a = vocab.encode(([6, 130, 70, 8, 27, 18, 8, 41, 280, 7, 287]))
print(a)
# b = vocab.index2word
# print(b)