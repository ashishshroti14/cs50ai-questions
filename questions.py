import nltk
import sys
import os
import string
import math
from collections import Counter


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    documents = dict()

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), encoding="utf8") as file:
            contents = file.read()
            
        documents[filename] = contents

    return documents


    # raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    updated_document = document.lower()
    updated_document = nltk.word_tokenize(updated_document)

    updated_document = [word for word in updated_document if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english")]


    return updated_document
    # raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    e = math.e
    total_number_of_documents = len(documents)
    counts = dict()
    idfs  = dict()

    for document in documents:
        for word in documents[document]:
            if word not in counts:
                counts[word] = 0
    
    for word in counts:
        for document in documents:
            
            if word in documents[document]:

                counts[word] +=1
    
    for word in counts:
        if counts[word] != 0:

            idfs[word] = math.log( (float(total_number_of_documents)/counts[word]), e)

    return idfs




def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    idfs_ranked_documents = {}
    for query_word in query:
        if query_word in idfs:


            query_word_tfidfs = {}
            for document in files:
                query_word_count = len([i for i in files[document] if i == query_word])
                tfidf = query_word_count * idfs[query_word]
                query_word_tfidfs[document] = tfidf

            idfs_ranked_documents = dict(Counter(idfs_ranked_documents) + Counter(query_word_tfidfs))

    my_top_files = list(sorted(idfs_ranked_documents.items(), key=lambda x: x[1], reverse=True)[:n])
    my_top_files = [file_tuple[0] for file_tuple in my_top_files]
    return my_top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """


    idfs_ranked_sentences = {}
    query_term_densities = {}
    for query_word in query:
        if query_word in idfs:


            query_word_idfs = {}
            for sentence in sentences:
                number_of_query_words = len([i for i in query if i in sentences[sentence]])
                query_term_densities[sentence] = float(number_of_query_words)/len(sentences[sentence])
                if query_word in sentences[sentence]:
                    idf =  idfs[query_word]
                    query_word_idfs[sentence] = idf

            idfs_ranked_sentences = dict(Counter(idfs_ranked_sentences) + Counter(query_word_idfs))



    my_top_sentences = list(sorted(idfs_ranked_sentences.items(), key=lambda x: (x[1], query_term_densities[x[0]]),   reverse=True)[:n])

    my_top_sentences = [file_tuple[0] for file_tuple in my_top_sentences]
   
    return my_top_sentences





if __name__ == "__main__":
    main()
