import tensorflow as tf

    # Compute vocabulary length
    vocabulary_size = len(vocab_processor.vocabulary_)
    print ('vocabulary_size=' + str(vocabulary_size))

    # Create embedding marix
    embeddings = init_multiple_pretrained_embeddings(vocab_processor.vocabulary_, embedding_paths, params['word_embedding_size'])

    # Split training set to two sets of word indices NUMPY matrices and labels list
    train_1 = matrices[0][0]
    train_2 = matrices[0][1]

    y_train = matrices[0][2]

    # Create dataset object for batching
    dataset = Dataset(train_1, train_2, y_train, params['batch_size'])

    # Split dev set to two sets of word indices NUMPY matrices and labels list
    dev_1 = matrices[1][0]
    dev_2 = matrices[1][1]
    dev_1_lengths = (dev_1 != 0).cumsum(1).argmax(1) + 1
    dev_2_lengths = (dev_2 != 0).cumsum(1).argmax(1) + 1
    y_dev = matrices[1][2]

    test_1 = matrices[2][0]
    test_2 = matrices[2][1]
    test_1_lengths = (test_1 != 0).cumsum(1).argmax(1) + 1
    test_2_lengths = (test_2 != 0).cumsum(1).argmax(1) + 1
    y_test = matrices[2][2]


    # define placeholders for first sentences, second sentences and labels
    # x_1 and x_2 are batches of sentences: 2D matrices of the shape [batch_size, max_document]
    x_1 = tf.placeholder(tf.int64, [None, None], name='x_1') # [None, params['max_document_length']]?
    x_2 = tf.placeholder(tf.int64, [None, None], name='x_2') # [None, params['max_document_length']]?
    x_1_lengths = tf.placeholder(tf.int64, [None], name='x_1_len')
    x_2_lengths = tf.placeholder(tf.int64, [None], name='x_2_len')
