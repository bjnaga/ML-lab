# Tokenizing using NLTK

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize


s = '''Good muffins cost $3.88\nin New York.  Please buy me
... two of them.\n\nThanks.'''

word_tokenize(s) 
print(word_tokenize(s) )
print(sent_tokenize(s))
# Filtering using stop word
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize

# nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(s)
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
# Stemming using nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize




ps = PorterStemmer()
 
# choose some words to be stemmed
words = ["program", "programs", "programmer", "programming", "programmers"]
 
for w in words:
    print(w, " : ", ps.stem(w))

# Parts of speech tagging 


s = '''Good muffins cost $3.88\nin New York.  Please buy me
... two of them.\n\nThanks.'''

word_tokens = word_tokenize(s) 

tagged = nltk.pos_tag(word_tokens)
print(tagged)    

import nltk

# https://www.nltk.org/api/nltk.tokenize.punkt.html
# nltk.download('punkt') # Punkt Sentence Tokenizer

# https://www.nltk.org/api/nltk.tag.perceptron.html
# nltk.download('averaged_perceptron_tagger') #  uses the perceptron algorithm to predict which POS-tag is most likely given the word.

from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize

# Example sentence
sentence = "Educative Answers is a free web encyclopedia written by devs for devs."

# # Tokenization
tokens = word_tokenize(sentence)

# # POS tagging
pos_tags = nltk.pos_tag(tokens)

# Chunking patterns
chunk_patterns = r"""
    NP: {<DT>?<JJ>*<NN>}  # Chunk noun phrases
    VP: {<VB.*><NP|PP>}  # Chunk verb phrases
"""


# chunking is a powerful tool for analyzing sentences and extracting meaningful noun and verb phrases by 
# grouping words together based on their grammar

# Create a chunk parser
chunk_parser = RegexpParser(chunk_patterns)

# Perform chunking
result = chunk_parser.parse(pos_tags)

# Print the chunked result
print(result)

namedEnt = nltk.ne_chunk(pos_tags, binary=False)
namedEnt.draw()


