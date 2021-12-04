import  tensorflow as tf

tf.keras.backend.clear_session()

# Hyper-parameters
D_MODEL = 128 # 512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROPOUT = 0.1 # 0.1
BATCH_SIZE = 64
BUFFER_SIZE = 20000
MAX_LENGTH = 100
EPOCHS = 10

#load data paths
file_path='data/combined-newsqa-data-v1.json'
write_path = 'data/qa_data.csv'
sample_data_path = 'data/sample_data.csv'