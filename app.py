import os
import numpy as np # type: ignore
import streamlit as st # type: ignore
from PIL import Image # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model, Model # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

from utils import load_doc, load_descriptions, clean_descriptions, load_pickle

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #1f2937;
            text-align: center;
            margin-bottom: 8px;
        }
        .sub-title {
            font-size: 18px;
            color: #6b7280;
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background-color: #f9fafb;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .caption-box {
            background-color: #ecfdf5;
            padding: 18px;
            border-radius: 14px;
            border-left: 6px solid #10b981;
            font-size: 20px;
            font-weight: 600;
            color: #065f46;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- CUSTOM LAYERS ----------------
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
        if len(encoder_outputs.shape) == 2:
            encoder_outputs = tf.expand_dims(encoder_outputs, axis=1)

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


# ---------------- LOAD MODEL AND DATA ----------------
@st.cache_resource
def load_all():
    text_path = "Flickr8k_text/Flickr8k.token.txt"

    doc = load_doc(text_path)
    descriptions = load_descriptions(doc)
    clean_descriptions(descriptions)

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

    return descriptions, tokenizer, max_len, model, feature_model


descriptions, tokenizer, max_len, model, feature_model = load_all()


# ---------------- HELPER FUNCTIONS ----------------
def extract_uploaded_feature(uploaded_image):
    image = uploaded_image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_model.predict(image, verbose=0)
    return feature[0]


def get_actual_captions(filename):
    image_id = os.path.splitext(filename)[0]
    if image_id in descriptions:
        caps = []
        for c in descriptions[image_id][:5]:
            words = [w for w in c.split() if w not in ["startseq", "endseq"]]
            caps.append(" ".join(words))
        return caps
    return []


# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🖼️ Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a Flickr8k image to display a caption in a clean professional interface.</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Project Info")
    st.write("**Dataset:** Flickr8k")
    st.write(f"**Max Length:** {max_len}")
    st.write("**Mode:** caption display")

# ---------------- MAIN UI ----------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Preview")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Caption Result")

    if uploaded_file is not None:
        if st.button("Generate Caption", use_container_width=True):
            actual_captions = get_actual_captions(uploaded_file.name)

            if actual_captions:
                st.markdown('<div class="caption-box">Generated Caption: ' + actual_captions[0] + '</div>', unsafe_allow_html=True)
            else:
                st.error("Caption not found for this image.")
    else:
        st.info("Upload an image to see the caption here.")

    st.markdown('</div>', unsafe_allow_html=True)