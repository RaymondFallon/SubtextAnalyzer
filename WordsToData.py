import gensim
import pickle


vocab_size = 250000

subtext = 'violent'

print("Running for ", subtext, " with ", vocab_size, " words.")

dataFile = './ReadingSamples/' + subtext + '.txt'
newFile = './ReadingSamples_Converted/' + subtext + str(vocab_size) +'.txt'

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
        if w in index2word:
            indexedList.append(word2index[w])
        else:
            indexedList.append(0)
readfile.close()


print("Writing indexes to file...")
pickle.dump(indexedList, writefile)
writefile.close()

print("Word Count: ", len(indexedList))