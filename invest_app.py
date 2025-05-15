import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_models(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #quantization_config=bnb_config,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = 128001
    return model, tokenizer

@st.cache_resource
def load_embedding_model(embedding_model_name="all-MiniLM-L12-v2"):
    return SentenceTransformer(embedding_model_name)

@st.cache_data
def load_data():
    with open("clean_text.pkl", "rb") as f:
        cleaned_text = pickle.load(f)
    embeddings = torch.load("invest_embeddings.pt").to(device)
    return cleaned_text, embeddings

model, tokenizer = load_models()
embedding_model = load_embedding_model()
cleaned_text, embeddings = load_data()

def RAG_INVEST(query):
    query_encoded = embedding_model.encode([query], convert_to_tensor=True)
    similarities = embedding_model.similarity(query_encoded, embeddings)
    scores, top_5_indices = torch.topk(similarities[0], k=5)

    CONTEXT_TEXT = '\n'.join([cleaned_text[idx] for idx in top_5_indices if similarities[0][idx] > 0])

    conversation = [
        {"role": "user", "content": f'''{query}
You are a knowledgeable investment assistant. You only answer using the provided context, which contains educational content about investing and stock markets.
If the question is unrelated to investments, say "This question is not related to investment education."
You must provide accurate, beginner-friendly explanations that a high school student can understand.
Use only the context below when answering:
CONTEXT: {CONTEXT_TEXT}
Answer in plain English within 200 words.'''},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=256
        )

    processed_text = tokenizer.decode(output[0][len(inputs.input_ids[0]) + 3:], skip_special_tokens=True)
    return processed_text, CONTEXT_TEXT

# Streamlit UI
st.title("📈 Investment Q&A Chatbot")

query = st.text_area("💬 Ask your investment question:", "What is the difference between stocks and mutual funds?")

if st.button("Get Answer") and query:
    with st.spinner("Searching through investment knowledge..."):
        response, CONTEXT_TEXT = RAG_INVEST(query)
    st.write(response)
    with st.expander("🔎 See Retrieved Context"):
        st.markdown(
            f'<div style="background-color:#f3f3f3; padding:10px; border-radius:5px;">{CONTEXT_TEXT}</div>',
            unsafe_allow_html=True
        )