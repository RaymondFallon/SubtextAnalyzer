import gensim
import re
from scipy import spatial

vocab_size = 50000
user_input = raw_input("Please type in the text that you would like to evaluate.")
# user_input = "If you really pressed me to come to an answer, it'd be really hard but " \
#              "I'd want to give it to you, so I would say: yes, yes, yes."

filename = './GoogleNews-vectors-negative300.bin'
google_model = gensim.models.KeyedVectors.load_word2vec_format(
    filename, binary=True, limit=vocab_size)

subtexts = ["no_subtext", "violent", "depressive", "sexual"]
distances = [0, 0, 0, 0]

for i in range(4):
    if i == 0:
        print("Analyzing your language...")
    else:
        print("Checking for any " + subtexts[i] + " subtext...")
    st_file = './New_Embeddings/' + subtexts[i] + str(vocab_size)
    st_model = gensim.models.KeyedVectors.load_word2vec_format(
        st_file, limit=vocab_size)
    for word in user_input.split():
        word = re.sub(r"[.,!?:$\"]*", "", word)
        if google_model.vocab.has_key(word):
            dist = spatial.distance.cosine(google_model.word_vec(word),
                                           st_model.word_vec(word))
            distances[i] += abs(dist)
ans = 3
for i in range(3):
    if distances[i] > distances[ans]:
        ans = i
ans_words = ["no particular", "some violent", "some depressive", "some sexual"]
print("I believe this sentence has " + ans_words[ans] + " undertones.")
