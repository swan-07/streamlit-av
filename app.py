import nltk
# import spacy
# from spacy.cli import download

# download("en_core_web_sm")

# nlp = spacy.load("en_core_web_sm")

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('conll2000')

import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import os
import pickle
from features import prepare_entry


@st.cache_resource
def load_models():
    snapshot_dir = snapshot_download(repo_id="swan07/final-models", repo_type="dataset")    
    model_path = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    
    feature_model_path = os.path.join(snapshot_dir, "featuremodel.p")
    with open(feature_model_path, 'rb') as f:
      clf, transformer, scaler, secondary_scaler = pickle.load(f)

    sentence_transformer = SentenceTransformer(model_path)

    
    logreg_model_path = os.path.join(snapshot_dir, "logreg.pkl")
    logreg_model = joblib.load(logreg_model_path)

    return sentence_transformer, tokenizer, model, clf, transformer, scaler, secondary_scaler, logreg_model

sentence_transformer, tokenizer, model, clf, transformer, scaler, secondary_scaler, logreg_model = load_models()

def get_attention_weights(model, tokenizer, text1, text2):
    inputs = tokenizer(text1, text2, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    attentions = outputs.attentions  # shape: (num_layers, batch_size, num_heads, seq_length, seq_length)

    attention = attentions[-1][0]  # shape: (num_heads, seq_length, seq_length)
    # attention = torch.stack(attentions).mean(dim=1).mean(dim=1).squeeze().detach().cpu().numpy()  # Shape: (seq_length, seq_length)

    # avg attention weights across all heads
    attention = attention.mean(dim=0).detach().cpu().numpy()  # shape: (seq_length, seq_length)
    
    # avg attention weights across all tokens for each token
    token_attention = attention.mean(axis=0)  # shape: (seq_length,)

    # normalize the attention weights
    attention = (token_attention - token_attention.min()) / (token_attention.max() - token_attention.min())
    
    # get tokenized input tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    filtered_tokens = []
    filtered_attention = []
    for token, att in zip(tokens, attention):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            filtered_tokens.append(token)
            filtered_attention.append(att)

    return filtered_tokens, filtered_attention

# highlight text based on attention weights
def highlight_text(tokens, attention_weights):
    html_text = ""
    for token, weight in zip(tokens, attention_weights):
        color = f"rgba(255, 0, 0, {weight})"
        html_text += f"<span style='background-color: {color}'>{token}</span> "
    return html_text

# get similarity probability
def get_similarity_probability(model, logreg_model, text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True).unsqueeze(0)
    embedding2 = model.encode(text2, convert_to_tensor=True).unsqueeze(0)
    
    cosine_score = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    
    # use the logreg model to get the probability
    probability = logreg_model.predict_proba(np.array([[cosine_score]]))[0][1]
    
    return probability


def get_top_features(differences, coef, feature_names, top_n=10):
    importances = np.abs(differences * coef)
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [(feature_names[i], float(importances[i])) for i in top_indices]
    return top_features

def process_single_entry(transformer, scaler, secondary_scaler, clf, preprocessed_doc1, preprocessed_doc2, all_feature_names):
    try:
        X1 = np.asarray(transformer.transform([preprocessed_doc1]).todense())
        X2 = np.asarray(transformer.transform([preprocessed_doc2]).todense())
        
        # Scale the data
        X1 = scaler.transform(X1)
        X2 = scaler.transform(X2)
        
        # Calculate the absolute difference and apply secondary scaling
        X = secondary_scaler.transform(np.abs(X1 - X2))
        
        # Predict the probability
        prob = clf.predict_proba(X)[0, 1]

        # Get top features
        top_features = get_top_features(np.abs(X1 - X2).flatten(), clf.coef_.flatten(), all_feature_names, top_n=10)

    except Exception as e:
        print('Exception predicting:', e)
        prob = 0.5
        top_features = []

    return float(prob), top_features

def get_all_feature_names(feature_union):
    feature_names = []
    for name, transformer in feature_union.transformer_list:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.append(name)
    return np.array(feature_names)

# STREAMLIT APP
st.title("Transparent Authorship Verification")

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("Enter the first text:", "")

with col2:
    text2 = st.text_area("Enter the second text:", "")

button_col1, button_col2 = st.columns(2)


bert_clicked = False
feature_vectors_clicked = False
with button_col1:
  if st.button("Go (Embedding)"):

      #load tokenizer and BERT model
      bert_clicked = True

      
      #get attention weights and highlight text
      tokens, attention = get_attention_weights(model, tokenizer, text1, text2)
      highlighted_text = highlight_text(tokens, attention)
      
      #get similarity probability
      probability = get_similarity_probability(sentence_transformer, logreg_model, text1, text2)
      
      st.session_state['highlighted_text'] = highlighted_text
      st.session_state['bert_probability'] = probability

with button_col2:
  if st.button("Go (Feature Vector)"):
      feature_vectors_clicked = True
      preprocessed_doc1 = prepare_entry(text1, mode='accurate', tokenizer='casual')
      preprocessed_doc2 = prepare_entry(text2, mode='accurate', tokenizer='casual')
      all_feature_names = get_all_feature_names(transformer)
      prob, top_features = process_single_entry(transformer, scaler, secondary_scaler, clf, preprocessed_doc1, preprocessed_doc2, all_feature_names)

      st.session_state['top_features'] = top_features
      st.session_state['feature_vectors_probability'] = prob

output_col1, output_col2 = st.columns(2)   
      
with output_col1:
    st.write("### Attention Highlighting")
    if 'highlighted_text' in st.session_state:
        st.markdown(f"<div style='border: 1px solid; padding: 10px;'>{st.session_state['highlighted_text']}</div>", unsafe_allow_html=True)

    st.write("### Probability Same Author (Embedding)")
    if 'bert_probability' in st.session_state:
        st.markdown(f"<div style='border: 1px solid; padding: 10px;'>{st.session_state['bert_probability']:.4f}</div>", unsafe_allow_html=True)

with output_col2:
    st.write("### Top 10 Features")
    if 'top_features' in st.session_state:
        feature_list = "".join([f"<li>{feature}: {importance:.4f}</li>" for feature, importance in st.session_state['top_features']])
        st.markdown(f"<div style='border: 1px solid; padding: 10px;'><ul>{feature_list}</ul></div>", unsafe_allow_html=True)

    st.write("### Probability Same Author (Feature Vector)")
    if 'feature_vectors_probability' in st.session_state:
        st.markdown(f"<div style='border: 1px solid; padding: 10px;'>{st.session_state['feature_vectors_probability']:.4f}</div>", unsafe_allow_html=True)

st.write("**Disclaimer:** Use these results at your own risk. Models may give inaccurate results.")
