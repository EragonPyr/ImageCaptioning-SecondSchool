import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, RepeatVector, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

embedding_dim = 256
units = 512
max_length = 34
vocab_size = 5000


def load_inception_model():
    inception_model = InceptionV3(include_top=False, weights='imagenet')
    new_input = inception_model.input
    hidden_layer = inception_model.layers[-1].output
    pooling_layer = GlobalAveragePooling2D()(hidden_layer)
    return Model(inputs=new_input, outputs=pooling_layer)

image_features_extract_model = load_inception_model()


def load_captions(file_path):
    df = pd.read_csv(file_path)
    captions = df.groupby('image')['caption'].apply(list).to_dict()
    return captions


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_image_features(directory):
    features = {}
    print("Image extraction started\n")
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = preprocess_image(img_path)
        feature = image_features_extract_model.predict(img, verbose=0)
        features[img_name] = feature
    print("Image extraction finished\n")
    return features

def tokenize_captions(captions):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts([caption for captions_list in captions.values() for caption in captions_list])
    return tokenizer

def create_sequences(tokenizer, max_length, captions, features):
    X1, X2, y = [], [], []
    for key, caption_list in captions.items():
        print("Create sequence step: " + key + "\n")
        feature = features[key][0]
        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length):
    inputs1 = tf.keras.Input(shape=(2048,))
    fe1 = Dense(embedding_dim, activation='relu')(inputs1)
    fe2 = RepeatVector(max_length)(fe1)

    inputs2 = tf.keras.Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = LSTM(units, return_sequences=True)(se1)

    decoder1 = concatenate([fe2, se2])
    decoder2 = LSTM(units)(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model(image_dir, captions_file, model_save_path, tokenizer_save_path):
    captions = load_captions(captions_file)
    image_features = extract_image_features(image_dir)
    tokenizer = tokenize_captions(captions)
    X1, X2, y = create_sequences(tokenizer, max_length, captions, image_features)
    model = define_model(vocab_size, max_length)

    checkpoint = ModelCheckpoint(model_save_path, monitor='loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=3)

    model.fit([X1, X2], y, epochs=12, verbose=2, callbacks=[checkpoint, early_stopping])

    print(f'Saving tokenizer to {tokenizer_save_path}')
    with open(tokenizer_save_path, 'wb') as f:
        pickle.dump(tokenizer, f)

train_model('Flickr8k/Images/', 'Flickr8k/captions.txt', 'Resources/image_captioning_model.keras', 'Resources/tokenizer.pkl')

input("Training complete. Press Enter to exit.")