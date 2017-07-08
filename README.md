# Classification-of-poetry-by-author-using-RNN

Here machine learning is used to train a model to understand the different styles of writing of poets.
Robert Frost and Edgar Allen poems are used to train the model to predict which poet has written a given sentence. 
 
To do this, I used the python Natural Language Took Kit(nltk) to get the parts of speech tags of every sentence. The one-hot-encoded vector of the POS tags forms the input to the recurrent model. The output this time will not be a sequence but only the only the final value which will give the probability of the sequence written by a particular poet.
