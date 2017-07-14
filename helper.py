import collections
from time import time
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def words_to_index(data, column, max_words=None, length=None):
    """Translates words into integers representing their frequency rank.
    
    Parameters:
    data -- A Pandas DataFrame.
    column -- The label of the column to process.
    max_words -- The maximum number of most common words to consider. A value of None means all words will be considered.
    length -- Make each entry a fixed length through zero-padding or truncation. None does not fix the length.
    
    Returns:
    new_data -- A list of lists containing the processed data.
    index -- A dict mapping words to their index.
    reverse_index -- A dict mapping indices to words.
    """
    # First tokenize the entire corpus of words into a flattened list
    words = []
    song_indices = [0]
    last_index = 0
    print("Tokenizing...")
    start = time()
    for sample in data[column].values:
        tokens = text_to_word_sequence(sample)
        song_indices.append(len(tokens) + last_index)
        words.extend(tokens)
        last_index += len(tokens)
    print("Finished tokenizing after {:.3f}s".format(time() - start))
    
    # Count each word, then build an index out of the most common words
    count = [("NULL", -1)]
    count.extend(collections.Counter(words).most_common(max_words - 1 if max_words != None else None))
    index = {}
    for word, _ in count:
        index[word] = len(index)
    reverse_index = dict(zip(index.values(), index.keys()))
        
    # Convert the words in each entry into their numerical representations using the index
    new_data = []
    print("Converting words to indices...")
    start = time()
    for i in range(data.shape[0]):
        new_sample = []
        sample = words[song_indices[i]:song_indices[i+1]]
        for word in sample:
            if word in index:
                new_sample.append(index[word])
            else:
                new_sample.append(0)
        new_data.append(new_sample)
    print("Finished after {:.3f}s".format(time() - start))
    
    # Pad or truncate each entry
    if length != None:
        new_data = pad_sequences(new_data, length, padding="post", truncating="post")
            
    return new_data, index, reverse_index