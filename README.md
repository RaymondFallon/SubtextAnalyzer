Subtext Analyzer

The `SubtextAnalyzer` module is a program designed to take as input a small sample of 
English language and decide if it contains any subtext.  At present, there are only 
four types of potential output: `violent`, `sexual`, `depressive`, or `no_subtext`.  
It functions by loading the pre-trained Word2Vec embeddings by Google, then copying
and re-training those embeddings in the different directions of each viable subtext.
The result is 4 sets of new embeddings to compare against the original Google embeddings.


The program functions in three parts:

## `WordsToData.py` 
`WordsToData` takes as input a large text file of English language which is deemed to 
contain subtext of a particular type, let's say `depressive`, as well as a vocabulary 
size, `vocab_size`. We will then use Google's pre-trained embeddings and take the most
used `vocab_size` number of words as our vocabulary. `WordsToData.py` will then create 
a new file, where each word is replaced with the corresponding index of that word.  If
the word from the input file is not in our vocabulary, it will instead be replaced by a
0. 

#### What's lost?
This does not take into account multi-word phrases like "New_Orleans" or "total_recall," 
many of which have been helpfully captured as single words in Google's model.  

## `SubtextRetraining.py`
`SubtextRetraining` is a TensorFlow program that retrains the Google word embeddings on
the file that `WordsToData` has produced.  It uses a [skip-gram](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) model and 
[noise-contrastive estimation](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) sampling to create the new embeddings for each subtext.

## `SubtextAnalyzer.py`
Finally, `SubtextAnalyzer` will take a string of English language and break it down into
individual words.  Then each word will be tested against the original word embeddings and 
all of the new subtext-specific embeddings.  Specifically, for each word and each subtext,
we're looking at the cosine-distance between the vector representation of that word in its
original Google-trained form and its new subtext-specific form.  Which-ever subtext has the
greatest cumulative distance for all the input words is chosen as the subtext of the input
and is returned as the answer.  

#### Confessions and Defense
As it stands, the program has a rather low success rate.  I believe this is due, in large
part, to the relatively small sample text I have for each subtext.  The input I used had 
about 5,000 - 20,000 words per subtext and is pulled from very varied sources. At present, 
the results are based more on the principle of "was this word ever re-trained for this
subtext? If so, then the co-sine distance is quite large."  For `SubtextAnaylzer` to work 
properly, the question needs to be "<i>when</i> this word was retrained for this particular
subtext, how different did it become from its original embedding?"  
