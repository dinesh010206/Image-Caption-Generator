import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore
from nltk.translate.bleu_score import corpus_bleu # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model, Model # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

from utils import (
    load_doc, load_descriptions, clean_descriptions, load_set,
    load_clean_descriptions, load_pickle
)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.att2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_outputs, training=False):
        attn1 = self.att1(query=x, value=x, key=x)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        attn2 = self.att2(query=out1, value=encoder_outputs, key=encoder_outputs)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

        ffn_output = self.ffn(out2)
        return self.layernorm3(out2 + self.dropout3(ffn_output, training=training))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config


TEXT_PATH = "Flickr8k_text/Flickr8k.token.txt"
TEST_PATH = "Flickr8k_text/Flickr_8k.testImages.txt"
IMAGE_FOLDER = "Flicker8k_Dataset"

doc = load_doc(TEXT_PATH)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)

test = load_set(TEST_PATH)
test_descriptions = load_clean_descriptions(descriptions, test)

print("Image folder exists:", os.path.exists(IMAGE_FOLDER))
print("Total test image ids:", len(test))
print("Total test descriptions loaded:", len(test_descriptions))
print("Sample test ids:", list(test_descriptions.keys())[:5])

data = load_pickle("model/tokenizer.pkl")
tokenizer = data["tokenizer"]
max_len = data["max_len"]

model = load_model(
    "model/best_model.keras",
    compile=False,
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock
    }
)

base_model = ResNet50(weights="imagenet")
feature_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)


def extract_single_feature(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_model.predict(image, verbose=0)
    return feature[0]


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo, max_len):
    in_text = ["startseq"]

    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        sequence = sequence + [0] * (max_len - len(sequence))
        sequence = np.array([sequence[:max_len]])

        yhat = model.predict(
            {"image_input": np.array([photo]), "text_input": sequence},
            verbose=0
        )
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)
        if word is None:
            break

        in_text.append(word)

        if word == "endseq":
            break

    final = [w for w in in_text if w not in ["startseq", "endseq"]]
    return final


actual, predicted = [], []

count = 0
for image_id, desc_list in test_descriptions.items():
    count += 1
    image_path = os.path.join(IMAGE_FOLDER, f"{image_id}.jpg")

    if not os.path.exists(image_path):
        continue

    photo = extract_single_feature(image_path)
    yhat = generate_caption(model, tokenizer, photo, max_len)
    references = [d.split()[1:-1] for d in desc_list]

    actual.append(references)
    predicted.append(yhat)

   
    print("Actual generated Captions:")
    for ref in references[:3]:
        print(" -", " ".join(ref))

    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Actual generated Caption: " + " ".join(references[0]), fontsize=12)
    plt.show()

    if count == 10:
        break

   

print("\nEvaluation Results:")
print("Number of evaluated samples:", len(actual))
print("Number of predicted samples:", len(predicted))

if len(actual) == 0 or len(predicted) == 0:
    print("No valid samples available for BLEU calculation.")
    print("Check image paths, test dataset loading, and caption generation.")
else:
    print("BLEU-1:", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2:", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print("BLEU-3:", corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print("BLEU-4:", corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))