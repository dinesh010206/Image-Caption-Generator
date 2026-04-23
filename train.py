import os
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Dense, Dropout, Embedding, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D, RepeatVector
)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from utils import (
    load_doc, load_descriptions, clean_descriptions, load_set,
    load_clean_descriptions, create_tokenizer, max_length, vocab_size,
    extract_features, save_pickle, load_pickle, data_generator
)

IMAGES_PATH = "Flicker8k_Dataset"
TEXT_PATH = "Flickr8k_text/Flickr8k.token.txt"
TRAIN_PATH = "Flickr8k_text/Flickr_8k.trainImages.txt"

doc = load_doc(TEXT_PATH)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)

train = load_set(TRAIN_PATH)
train_descriptions = load_clean_descriptions(descriptions, train)

tokenizer = create_tokenizer(train_descriptions)
vocab = vocab_size(tokenizer)
max_len = max_length(train_descriptions)

print("Vocabulary Size:", vocab)
print("Max Length:", max_len)

save_pickle({"tokenizer": tokenizer, "max_len": max_len}, "model/tokenizer.pkl")

if not os.path.exists("features/features.pkl"):
    print("Extracting image features...")
    features = extract_features(IMAGES_PATH)
    save_pickle(features, "features/features.pkl")
else:
    features = load_pickle("features/features.pkl")

train_features = {k: features[k] for k in train if k in features}

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, x, encoder_outputs, training=False):
        attn1 = self.att1(query=x, value=x, key=x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.att2(query=out1, value=encoder_outputs, key=encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

def define_model(vocab_size, max_len, embed_dim=256, num_heads=4, ff_dim=512):
    image_input = Input(shape=(2048,), name="image_input")
    img_dense = Dense(embed_dim, activation="relu")(image_input)
    img_dropout = Dropout(0.4)(img_dense)

    encoder_outputs = RepeatVector(max_len)(img_dropout)

    text_input = Input(shape=(max_len,), name="text_input")
    x = PositionalEmbedding(max_len, vocab_size, embed_dim)(text_input)

    decoder_block = TransformerDecoderBlock(embed_dim, num_heads, ff_dim)
    x = decoder_block(x, encoder_outputs)

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        query=x, value=encoder_outputs, key=encoder_outputs
    )
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    output = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
    return model

model = define_model(vocab, max_len)
model.summary()

checkpoint = ModelCheckpoint(
    "model/best_model.keras",
    monitor="loss",
    verbose=1,
    mode="min"
)

epochs = 10
batch_size = 32
steps = 1000

generator = data_generator(
    train_descriptions, train_features, tokenizer, max_len, vocab, batch_size
)

model.fit(
    generator,
    epochs=epochs,
    steps_per_epoch=steps,
    callbacks=[checkpoint],
    verbose=1
)