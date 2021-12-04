import tensorflow as tf
from tensorflow.python.keras.backend import dtype
FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
LOWER_CASE = True
MAX_LEN = 300
EMBED_SIZE = 200
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.layers import SpatialDropout1D, Dense, Dropout, Input, concatenate, Embedding
from sklearn.preprocessing import LabelEncoder


import numpy as np

def define_tokenizer(q_sent,p_sent):
    sentences = p_sent+q_sent
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters=FILTERS,
        lower=LOWER_CASE
    )
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def encode(sentences, tokenizer):
    encoded_sentences = tokenizer.texts_to_sequences(sentences)

    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_sentences,
        padding='post',
        maxlen=MAX_LEN
    )
    return(encoded_sentences)


def load_glove(path_embedding):
    embedding_dict = {}

    with open(path_embedding, 'r',encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vectors

    f.close()
    return(embedding_dict)

def get_embedding(tokenizer,emb_dim,embedding_dict):
    num_words = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((num_words, emb_dim))

    for word, i in tokenizer.word_index.items():
        if i > num_words:
            continue

        emb_vec = embedding_dict.get(word)

        if emb_vec is not None:
            embedding_matrix[i] = emb_vec
    return(embedding_matrix)

def lvl_encoding(y):
    lc = LabelEncoder()
    y = lc.fit_transform(y)
    return(y)

def build_model(tokenizer,emb_matrix,questons, paragraph,ans_st_index,nclass):

    

    print('Embedding Layer...')
    embedding = Embedding(
        len(tokenizer.word_index) + 1,
        EMBED_SIZE,
        embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
        trainable=False
    )

    print('Question Input Layer...')
    question_input = Input(shape=(MAX_LEN,))
    print(1)
    question_x =embedding(question_input)
    print(2)
    question_x = SpatialDropout1D(0.2)(question_x)
    print(3)
    question_x = Bidirectional(LSTM(20, return_sequences=True))(question_x)
    print(4)
    question_x = GlobalMaxPooling1D()(question_x)
    print(5)

    print('Answer Input Layer...')
    answer_input = Input(shape=(MAX_LEN,))
    answer_x = embedding(answer_input)
    answer_x = SpatialDropout1D(0.2)(answer_x)
    answer_x = Bidirectional(LSTM(20, return_sequences=True))(answer_x)
    answer_x = GlobalMaxPooling1D()(answer_x)

    print('Combine Q&A...')
    combined_x = concatenate([question_x, answer_x],axis=0)
    combined_x = Dense(10, activation='relu')(combined_x)
    combined_x = Dropout(0.5)(combined_x)
    combined_x = Dense(10, activation='relu')(combined_x)
    combined_x = Dropout(0.5)(combined_x)
    output = Dense(nclass, activation='softmax')(combined_x)

    # combine model parts into one
    model = tf.keras.models.Model(inputs=[answer_input, question_input], outputs=output)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")

    print('Model Compilation...')
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['Accuracy']
        
    )
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
    ]

    print('Model Fit Started...')
    history = model.fit(
        x=[questons, paragraph],
        y=ans_st_index,
        epochs=5,
        callbacks=callbacks,
        batch_size=32,
        shuffle=True,verbose=1)


def main(q_data,p_data,ans_st_index):
    print('Tokenizer Initiated')
    tokenizer = define_tokenizer(q_data,p_data)

    print('Question tokenizer running...')
    p_data_1 = encode(p_data, tokenizer)
    print('Paragraph tokenizer running...')
    q_data_1 = encode(q_data, tokenizer)

    ans_st_index = lvl_encoding(ans_st_index)

    questons, paragraph = np.array(q_data_1), np.array(p_data_1)
    #ans_st_index = np.array(ans_st_index).reshape(-1,1)
    # ans_st_index = [str(int(i)) for i in ans_st_index]
    n_class = len(np.unique(ans_st_index))

    y= tf.keras.utils.to_categorical(ans_st_index)

    print(questons.shape)
    print( paragraph.shape)
    print(y.shape)

    print('Embedding loaded...')

    emb =load_glove('data/glove.6B.200d.txt/glove.6B.200d.txt')

    print('Embedding Matrix created...')
    emb_mat = get_embedding(tokenizer, emb_dim=EMBED_SIZE, embedding_dict=emb)
    print('Model Running...')

    

    _= build_model(tokenizer=tokenizer, emb_matrix = emb_mat
                        , questons=questons, paragraph=paragraph
                        , ans_st_index=y,nclass=y.shape[-1])

    print('Model completed...')

