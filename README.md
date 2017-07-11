# Classification-of-poetry-by-author-using-RNN

Here objective is to train a model to identify the poet given a line.
I used Robert Frost and Edgar Allen poems as datasets to train the model to predict which poet has written a given line.

To do this, I used the python Natural Language Took Kit(nltk) to get the parts of speech tags of every sentence. The one-hot-encoded vector of the POS tags forms the input to the recurrent model. The best classification rate observed is 0.84.
