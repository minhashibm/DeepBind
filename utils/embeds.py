'''
This method constructs several input channels for a CNN.
Every input channel is a different encoding matrix.
Input:
        vocabulary      - TF Vocabulary object constructed on data
        embedding_paths - List of word embeddings files
        embedding_size  - The dimension of the word embeddings
Output:
        Multiple embedding matrix:
        Dimension I  : word
        Dimension II : real number embedding
        Dimension III: embedding type (Word2Vec, Glove, etc.)

        The UNK token is also included in the cube returned (as the firs row of every matrix).
        The last layer row of every embedding matrix is a constant 0 - for padding.
        The embeddings matrices are variables.
'''
def init_multiple_pretrained_embeddings(vocabulary, embedding_paths, embedding_size):

    # Construct the intersections of the embedding files
    common_words = set()
    for path_index in range(len(embedding_paths)):
        epath = embedding_paths[path_index]
        current_words = set()
        with open(epath) as f:
            for line in f:
                line = line.strip().split()
                word = line[0]
                current_words.add(word)
        # intersect current embedding words with previous (if there are)
        if path_index == 0:
            common_words = current_words
        else:
            common_words = common_words.intersection(current_words)

    # Construct a mapping of WORD_IDX => [EMBED1, EMBED2, ...]
    existing_embeddings = {}
    for embedding_path in embedding_paths:
        print ("Processing ", embedding_path)
        file = open(embedding_path, 'r')
        count = 0
        for i, line in enumerate(file.readlines()):
            if i % 10000 == 0:
                print ('iterations=', i)
            row = line.strip().split(' ')
            word = row[0]
            if word in vocabulary._mapping and word in common_words:
                count +=1
                embedding = row[1:]
                word_index = vocabulary._mapping[word]
                if word_index not in existing_embeddings:
                    existing_embeddings[word_index] = list()
                existing_embeddings[word_index].append(embedding)
