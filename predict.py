import os
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model, Model # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from utils import load_pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

        if len(encoder_outputs.shape) == 2:
            encoder_outputs = tf.expand_dims(encoder_outputs, axis=1)

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


# ---------------- LOAD ----------------
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

index_word = {index: word for word, index in tokenizer.word_index.items()}


# ---------------- HELPERS ----------------
def extract_single_feature(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_model.predict(image, verbose=0)
    return feature[0].astype("float32")


def clean_final_caption(words):
    weak_words = {
        "and", "is", "in", "on", "at", "of", "the", "a", "an", "to", "with",
        "for", "by", "from", "that", "this"
    }

    person_words = {"man", "woman", "girl", "boy", "child", "person", "people", "men", "women", "children"}

    cleaned = []
    for w in words:
        if w in ["startseq", "endseq"]:
            continue
        if len(cleaned) == 0 and w in weak_words:
            continue
        if cleaned and w == cleaned[-1]:
            continue
        if w in cleaned:
            continue
        cleaned.append(w)

    while cleaned and cleaned[-1] in weak_words:
        cleaned.pop()

    # ---- sentence correction ----
    if not cleaned:
        return "No caption generated"

    # if all words are person-related, make it natural
    if all(w in person_words for w in cleaned):
        if len(cleaned) == 1:
            return f"A {cleaned[0]}"
        elif len(cleaned) == 2:
            return f"A {cleaned[0]} and a {cleaned[1]}"
        else:
            return "A group of people"

    # if first 2 words are person words
    if len(cleaned) >= 2 and cleaned[0] in person_words and cleaned[1] in person_words:
        return f"A {cleaned[0]} and a {cleaned[1]}"

    # if only one word
    if len(cleaned) == 1:
        return f"A {cleaned[0]}"

    return " ".join(cleaned).capitalize()


def generate_caption_beam(model, tokenizer, photo, max_len, beam_width=7, min_words=3):
    start_id = tokenizer.word_index.get("startseq")
    end_id = tokenizer.word_index.get("endseq")

    if start_id is None or end_id is None:
        return "No caption generated"

    weak_words = {
        "and", "is", "in", "on", "at", "of", "the", "a", "an", "to", "with",
        "for", "by", "from", "that", "this"
    }

    sequences = [([start_id], 0.0)]

    for step in range(max_len):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end_id:
                all_candidates.append((seq, score))
                continue

            padded = seq + [0] * (max_len - len(seq))
            seq_input = np.array([padded[:max_len]], dtype=np.int32)
            photo_input = np.array([photo], dtype=np.float32)

            preds = model.predict(
                {"image_input": photo_input, "text_input": seq_input},
                verbose=0
            )[0]

            top_ids = np.argsort(preds)[-30:][::-1]

            for idx in top_ids:
                idx = int(idx)
                word = index_word.get(idx)

                if word is None:
                    continue

                current_words = [
                    index_word.get(i) for i in seq
                    if i in index_word and index_word.get(i) not in ["startseq", "endseq"]
                ]

                if idx == end_id and len(current_words) < min_words:
                    continue

                if len(current_words) < 2 and word in weak_words:
                    continue

                if len(current_words) > 0 and word == current_words[-1]:
                    continue

                if word in current_words:
                    continue

                penalty = 0.0
                if word in weak_words:
                    penalty = 2.0

                prob = float(preds[idx])
                new_score = score - np.log(prob + 1e-10) + penalty
                all_candidates.append((seq + [idx], new_score))

        if not all_candidates:
            break

        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]

        best_seq = sequences[0][0]
        best_words = [
            index_word.get(i) for i in best_seq
            if i in index_word and index_word.get(i) not in ["startseq", "endseq"]
        ]
        print(f"Step {step+1}: {' '.join(best_words)}")

        if best_seq[-1] == end_id:
            break

    best_seq = sequences[0][0]
    final_words = []

    for idx in best_seq:
        word = index_word.get(idx)
        if word is None or word in ["startseq", "endseq"]:
            continue
        final_words.append(word)

    return clean_final_caption(final_words)


# ---------------- TEST ----------------
image_path = "Flickr8k_Dataset/10815824_2997e03d76.jpg"

if not os.path.exists(image_path):
    print("Image not found:", image_path)
else:
    photo = extract_single_feature(image_path)
    caption = generate_caption_beam(model, tokenizer, photo, max_len, beam_width=7, min_words=3)
    print("Generated Caption:", caption)