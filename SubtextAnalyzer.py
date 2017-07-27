import gensim
from scipy import spatial

vocab_size = 250000
user_input = "If you really pressed me to come to an answer, it'd be really hard but " \
             "I'd want to give it to you, so I would say: yes, yes, yes."

filename = './GoogleNews-vectors-negative300.bin'
# Model is limited to 500,000 words at present due to memory constraints
# This should be properly run with no limit
google_model = gensim.models.KeyedVectors.load_word2vec_format(
    filename, binary=True, limit=vocab_size)

subtexts = ["no_subtext", "violent", "depressive", "sexual"]
distances = [0, 0, 0, 0]

for i in range(4):
    print("Loading ", subtexts[i])
    st_file = './New_Embeddings/' + subtexts[i] + str(vocab_size)
    st_model = gensim.models.KeyedVectors.load_word2vec_format(
        st_file, limit=vocab_size)
    print("Loaded.  Running...")
    for word in user_input.split():
        word = word.replace(",", "")
        word = word.replace(".", "")
        word = word.replace(":", "")
        if google_model.vocab.has_key(word):
            dist = spatial.distance.cosine(google_model.word_vec(word),
                                           st_model.word_vec(word))
            distances[i] += abs(dist)
            print(word, ": ", dist)
print(distances)
ans = 3
for i in range(3):
    if distances[i] > distances[ans]:
        ans = i
ans_words = ["no particular", "some violent", "some depressive", "some sexual"]
print("I believe this sentence has " + ans_words[ans] + " undertones.")







# print(google_model.doesnt_match("man woman boy tree girl".split()))
# print(google_model.doesnt_match("up down left right north".split()))
# print(google_model.most_similar_cosmul(positive="rock stone brick".split(),
#                                 negative="clay sand".split(),
#                                 topn=5))
# print(google_model.most_similar_cosmul(positive="prince king girl woman".split(),
#                                 negative="man boy".split(),
#                                 topn=5))
