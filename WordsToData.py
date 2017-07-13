import gensim


vocab_size = 150000

dataFile = './ReadingSamples/depressive.txt'
newFile = './ReadingSamples_Converted/depressive' + str(vocab_size) +'.txt'

embedFile = './GoogleNews-vectors-negative300.bin'

print("Loading Pre-trained Model...")
model = gensim.models.KeyedVectors.load_word2vec_format(
    embedFile, binary=True, limit=vocab_size)
word2index = {}
index2word = model.index2word
for i in range(vocab_size):
    word2index[index2word[i]] = i
del model

readfile = open(dataFile, mode='r')
writefile = open(newFile, mode='w')

indexedList = []
idx = 0
print("Reading Input File...")
for line in readfile:
    idx += 1
    line = line.split()
    for w in line:
        w = w.replace(',', '')
        w = w.replace(' ', '')
        w = w.replace('!', '')
        w = w.replace('.', '')
        w = w.replace(';', '')
        w = w.replace('"', '')
        # if idx>65:
        #     print(w)
        if w in index2word:
            indexedList.append(word2index[w])
        else:
            indexedList.append(0)


print("Writing indexes to file...")
writefile.write(str(indexedList))

print(len(indexedList))
print("word2index(The) ", word2index["The"])
print("word2index(the) ", word2index["the"])
print("word2index(Son) ", word2index["Son"])
print("word2index(mother) ", word2index["mother"])
print("word2index(said) ", word2index["said"])

print("number of words: " , len(index2word), len(word2index))
print("59075: " , index2word[59075])
print("1781: " , index2word[1781])
print("47296: " , index2word[47296])