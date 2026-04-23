import os
import string
import pickle
import numpy as np # type: ignore
from tqdm import tqdm # type: ignore

from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model # type: ignore

def load_doc(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_descriptions(doc):
    mapping = {}
    for line in doc.strip().split('\n'):
        tokens = line.split('\t')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0], tokens[1]
        image_id = image_id.split('.')[0]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.lower()
            desc = desc.split()
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = 'startseq ' + ' '.join(desc) + ' endseq'

def load_set(filename):
    doc = load_doc(filename)
    dataset = []
    for line in doc.strip().split('\n'):
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(descriptions, dataset):
    clean = {}
    for key, desc_list in descriptions.items():
        if key in dataset:
            clean[key] = desc_list
    return clean

def to_lines(descriptions):
    all_desc = []
    for key in descriptions:
        all_desc.extend(descriptions[key])
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer(oov_token="unk")
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def vocab_size(tokenizer):
    return len(tokenizer.word_index) + 1

def extract_features(directory):
    model = ResNet50(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}

    for name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, name)
        try:
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = name.split('.')[0]
            features[image_id] = feature[0]
        except Exception as e:
            print(f"Error processing {name}: {e}")

    return features

def create_sequences(tokenizer, max_len, desc_list, photo, vocab_size):
    X1, X2, y = [], [], []

    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]

        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = in_seq + [0] * (max_len - len(in_seq))
            in_seq = in_seq[:max_len]

            out_seq_onehot = np.zeros(vocab_size)
            out_seq_onehot[out_seq] = 1.0

            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq_onehot)

    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, photos, tokenizer, max_len, vocab_size, batch_size):
    while True:
        X1, X2, y = [], [], []

        for key, desc_list in descriptions.items():
            if key not in photos:
                continue

            photo = photos[key]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, max_len, desc_list, photo, vocab_size
            )

            for i in range(len(in_img)):
                X1.append(in_img[i])
                X2.append(in_seq[i])
                y.append(out_word[i])

                if len(X1) == batch_size:
                    yield (
                        {"image_input": np.array(X1), "text_input": np.array(X2)},
                        np.array(y)
                    )
                    X1, X2, y = [], [], []

def save_pickle(data, filename):
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)