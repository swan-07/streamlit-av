import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Authorship Verification", layout="wide", page_icon="🔍")

# Cache model loading
@st.cache_resource
def load_models():
    """Load BERT, stylometric, and ensemble models"""
    # Load BERT from HuggingFace (change to your username after uploading)
    # After uploading: huggingface-cli upload swan07/bert-authorship-verification ./models/bert-finetuned-authorship
    bert_model = SentenceTransformer('swan07/bert-authorship-verification')

    # Load small models from repo (these can be committed to git)
    with open('stylometric_pan_model.pkl', 'rb') as f:
        stylo_data = pickle.load(f)

    with open('ensemble_model.pkl', 'rb') as f:
        ensemble_data = pickle.load(f)

    return bert_model, stylo_data, ensemble_data

def extract_punctuation(text):
    return re.sub(r'[^\s!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)

def get_stylometric_features(text1, text2, stylo_data):
    """Extract all stylometric features with names"""
    char_vectorizers = stylo_data['char_vectorizers']
    word_vectorizer = stylo_data['word_vectorizer']
    punct_vectorizer = stylo_data['punct_vectorizer']

    features = []
    feature_details = []

    # Char n-gram similarities
    for n in [3, 4, 5]:
        v1 = char_vectorizers[n].transform([text1])
        v2 = char_vectorizers[n].transform([text2])
        sim = cosine_similarity(v1, v2)[0, 0]
        features.append(sim)
        feature_details.append({
            'name': f'Character {n}-grams',
            'value': sim,
            'description': f'Similarity of {n}-character patterns (e.g., "the", "ing", "tion")'
        })

    # Word n-gram similarity
    v1 = word_vectorizer.transform([text1])
    v2 = word_vectorizer.transform([text2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)
    feature_details.append({
        'name': 'Word patterns',
        'value': sim,
        'description': 'Similarity of word usage and common phrases'
    })

    # Punctuation
    p1 = extract_punctuation(text1)
    p2 = extract_punctuation(text2)
    v1 = punct_vectorizer.transform([p1])
    v2 = punct_vectorizer.transform([p2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)
    feature_details.append({
        'name': 'Punctuation style',
        'value': sim,
        'description': 'Similarity of punctuation patterns (commas, periods, etc.)'
    })

    # Length ratio
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    features.append(len_ratio)
    feature_details.append({
        'name': 'Length ratio',
        'value': len_ratio,
        'description': 'How similar the text lengths are'
    })

    # Sentence ratio
    sent1 = len(text1.split('.'))
    sent2 = len(text2.split('.'))
    sent_ratio = min(sent1, sent2) / max(sent1, sent2) if max(sent1, sent2) > 0 else 1.0
    features.append(sent_ratio)
    feature_details.append({
        'name': 'Sentence structure',
        'value': sent_ratio,
        'description': 'Similarity of sentence count and structure'
    })

    # Average word length
    words1 = text1.split()
    words2 = text2.split()
    avg_len1 = np.mean([len(w) for w in words1]) if words1 else 0
    avg_len2 = np.mean([len(w) for w in words2]) if words2 else 0
    len_sim = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2) if max(avg_len1, avg_len2) > 0 else 1.0
    features.append(len_sim)
    feature_details.append({
        'name': 'Word length',
        'value': len_sim,
        'description': 'Similarity of average word length'
    })

    return features, feature_details

def analyze_texts(text1, text2, bert_model, stylo_data, ensemble_data):
    """Analyze two texts and return predictions with interpretability"""

    # BERT analysis
    emb1 = bert_model.encode(text1, convert_to_numpy=True)
    emb2 = bert_model.encode(text2, convert_to_numpy=True)
    bert_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    bert_pred = "Same Author" if bert_score >= 0.5 else "Different Authors"
    bert_conf = bert_score if bert_score >= 0.5 else (1 - bert_score)

    # Stylometric analysis
    features, feature_details = get_stylometric_features(text1, text2, stylo_data)
    stylo_classifier = stylo_data['classifier']
    stylo_score = stylo_classifier.predict_proba([features])[0, 1]
    stylo_pred = "Same Author" if stylo_score >= 0.5 else "Different Authors"
    stylo_conf = stylo_score if stylo_score >= 0.5 else (1 - stylo_score)

    # Ensemble analysis
    ensemble_combiner = ensemble_data['combiner']
    X = np.array([[bert_score, stylo_score]])
    ensemble_prob = ensemble_combiner.predict_proba(X)[0, 1]
    ensemble_pred = "Same Author" if ensemble_prob >= 0.5 else "Different Authors"
    ensemble_conf = ensemble_prob if ensemble_prob >= 0.5 else (1 - ensemble_prob)

    # Ensemble weights
    bert_weight = ensemble_combiner.coef_[0][0]
    stylo_weight = ensemble_combiner.coef_[0][1]
    bias = ensemble_combiner.intercept_[0]

    return {
        'bert': {
            'score': bert_score,
            'prediction': bert_pred,
            'confidence': bert_conf
        },
        'stylometric': {
            'score': stylo_score,
            'prediction': stylo_pred,
            'confidence': stylo_conf,
            'features': feature_details
        },
        'ensemble': {
            'score': ensemble_prob,
            'prediction': ensemble_pred,
            'confidence': ensemble_conf,
            'weights': {
                'bert': bert_weight,
                'stylo': stylo_weight,
                'bias': bias
            }
        }
    }

# Main app
st.title("Transparent Authorship Verification")
st.markdown("### Determine if two texts were written by the same author")
st.markdown("**Powered by BERT + Stylometric Analysis** | Accuracy: 73.9% | AUC: 0.823")

# Load models
with st.spinner("Loading models..."):
    bert_model, stylo_data, ensemble_data = load_models()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system combines:
    - **BERT**: Deep semantic analysis (73.9% acc)
    - **Stylometric**: Writing style patterns (62.2% acc)
    - **Ensemble**: Combined model (73.9% acc, 0.823 AUC)

    ### How it works:
    1. Paste two text samples
    2. Get predictions from all models
    3. See detailed interpretability
    """)

    st.header("📊 Model Info")
    st.metric("Training Data", "50K pairs")
    st.metric("Test Accuracy", "73.9%")
    st.metric("AUC Score", "0.823")

    st.header("📚 Dataset")
    st.markdown("""
    Models trained on [swan07/authorship-verification](https://huggingface.co/datasets/swan07/authorship-verification)
    dataset (325K pairs), combining 12 sources:

    - **PAN Competition** (2011, 2013-15, 2020)
    - **Reuters50**, **IMDB62**, **Blog Corpus**
    - **arXiv**, **Victorian Era**, **BAWE**
    - **DarkReddit**

    **Primary Citations:**
    - Manolache et al. (2021) - [Transferring BERT Knowledge](https://arxiv.org/abs/2112.05125)
    - Kestemont et al. (2020) - [PAN 2020 Overview](http://ceur-ws.org/Vol-2696/paper_264.pdf)
    """)

# Example texts (from swan07/authorship-verification test set)
example_same = {
    'text1': "This paper studies a prototype of inverse obstacle scattering problems whose governing equation is the GPE equation in two dimensions. An explicit method to extract information about the location and shape of unknown obstacles from the far field operator with a fixed wave number is given. The method is based on: an explicit construction of a modification of Green's function via the Vekua transform and the study of the asymptotic behaviour; an explicit density in the Herglotz wave function that approximates the modification of Green's function in the bounded domain surrounding unknown obstacles; a system of inequalities derived from Factorization's formula of the far field operator.",
    'text2': "We consider an inverse source problem for the GPE equation in a bounded domain. The problem is to reconstruct the shape of the support of a source term from the Cauchy data on the boundary of the solution of the governing equation. We prove that if the shape is a polygon, one can calculate its support function from such data. An application to the inverse boundary value problem is also included."
}

example_diff = {
    'text1': "The paper describes combinatorial synthesis approach with interval multiset estimates of system elements for modeling, analysis, design, and improvement of a modular telemetry system. Morphological (modular) system design and improvement are considered as composition of the telemetry system elements (components) configuration. The solving process is based on Hierarchical Morphological Multicriteria Design (HMMD): (i) multicriteria selection of alternatives for system components, (ii) synthesis of the selected alternatives into a resultant combination (while taking into account quality of the alternatives above and their compatibility).",
    'text2': "This method is an evo-devo approach to generate arbitrary 2D or 3D shapes; as such, it belongs to the field of artificial embryology. In silico experiments have proved the effectiveness of the method in devo-evolving shapes of any kind and complexity (in terms of number of cells, number of colours, etc.), establishing its potential to generate the complexity typical of biological systems. Furthermore, it has also been shown how the underlying model of development is able to produce the artificial version of key biological phenomena such as embryogenesis, the presence of junk DNA, the phenomenon of ageing and the process of carcinogenesis."
}

# Initialize session state for text inputs
if 'text1_input' not in st.session_state:
    st.session_state.text1_input = ""
if 'text2_input' not in st.session_state:
    st.session_state.text2_input = ""

# Examples
st.markdown("---")
col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 2])
with col_ex1:
    if st.button("📝 Load Example: Same Author"):
        st.session_state.text1_input = example_same['text1']
        st.session_state.text2_input = example_same['text2']

with col_ex2:
    if st.button("📝 Load Example: Different Authors"):
        st.session_state.text1_input = example_diff['text1']
        st.session_state.text2_input = example_diff['text2']

with col_ex3:
    st.caption("*Example texts from swan07/authorship-verification test set (academic papers)*")

# Text input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Text 1")
    text1 = st.text_area("Enter first text:", height=200,
                         placeholder="Paste the first text here...",
                         help="Minimum 50 characters recommended",
                         key="text1_input")
    st.caption(f"Length: {len(text1)} characters")

with col2:
    st.subheader("📄 Text 2")
    text2 = st.text_area("Enter second text:", height=200,
                         placeholder="Paste the second text here...",
                         help="Minimum 50 characters recommended",
                         key="text2_input")
    st.caption(f"Length: {len(text2)} characters")

# Analyze button
st.markdown("---")
if st.button("🔍 Analyze Authorship", type="primary", use_container_width=True):
    if len(text1) < 50 or len(text2) < 50:
        st.error("⚠️ Please enter at least 50 characters for each text.")
    else:
        with st.spinner("Analyzing texts..."):
            results = analyze_texts(text1, text2, bert_model, stylo_data, ensemble_data)

        st.markdown("---")
        st.header("📊 Analysis Results")

        # Main prediction
        ensemble_pred = results['ensemble']['prediction']
        ensemble_conf = results['ensemble']['confidence']

        if ensemble_pred == "Same Author":
            st.success(f"## ✅ {ensemble_pred}")
            st.markdown(f"**Confidence:** {ensemble_conf*100:.1f}%")
        else:
            st.error(f"## ❌ {ensemble_pred}")
            st.markdown(f"**Confidence:** {ensemble_conf*100:.1f}%")

        # Model comparison
        st.markdown("---")
        st.subheader("🤖 Individual Model Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### BERT")
            st.metric("Prediction", results['bert']['prediction'])
            st.metric("Score", f"{results['bert']['score']:.3f}")
            st.metric("Confidence", f"{results['bert']['confidence']*100:.1f}%")

        with col2:
            st.markdown("### Stylometric")
            st.metric("Prediction", results['stylometric']['prediction'])
            st.metric("Score", f"{results['stylometric']['score']:.3f}")
            st.metric("Confidence", f"{results['stylometric']['confidence']*100:.1f}%")

        with col3:
            st.markdown("### Ensemble")
            st.metric("Prediction", results['ensemble']['prediction'])
            st.metric("Score", f"{results['ensemble']['score']:.3f}")
            st.metric("Confidence", f"{results['ensemble']['confidence']*100:.1f}%")

        # Interpretability section
        st.markdown("---")
        st.header("🔬 Interpretability & Feature Analysis")

        tab1, tab2, tab3 = st.tabs(["📈 Stylometric Features", "⚖️ Ensemble Weights", "📊 Score Distribution"])

        with tab1:
            st.subheader("Writing Style Features")
            st.markdown("These features capture different aspects of writing style:")

            # Feature importance chart
            feature_names = [f['name'] for f in results['stylometric']['features']]
            feature_values = [f['value'] for f in results['stylometric']['features']]

            fig = px.bar(
                x=feature_values,
                y=feature_names,
                orientation='h',
                labels={'x': 'Similarity Score', 'y': 'Feature'},
                title='Stylometric Feature Scores'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Feature descriptions
            st.markdown("#### Feature Details:")
            for feature in results['stylometric']['features']:
                with st.expander(f"{feature['name']}: {feature['value']:.3f}"):
                    st.markdown(f"**Score:** {feature['value']:.3f}")
                    st.markdown(f"**Description:** {feature['description']}")

                    # Color coding
                    if feature['value'] > 0.7:
                        st.success("🟢 High similarity - strong match")
                    elif feature['value'] > 0.4:
                        st.warning("🟡 Moderate similarity - inconclusive")
                    else:
                        st.error("🔴 Low similarity - likely different")

        with tab2:
            st.subheader("How the Ensemble Combines Models")

            bert_w = results['ensemble']['weights']['bert']
            stylo_w = results['ensemble']['weights']['stylo']

            st.markdown(f"""
            The ensemble uses **logistic regression** to optimally combine the two models:

            ```
            Ensemble Score = σ({bert_w:.3f} × BERT + {stylo_w:.3f} × Stylometric + {results['ensemble']['weights']['bias']:.3f})
            ```

            Where σ is the sigmoid function that converts to probability.
            """)

            # Weight visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['BERT', 'Stylometric'],
                    y=[abs(bert_w), abs(stylo_w)],
                    text=[f"{bert_w:.3f}", f"{stylo_w:.3f}"],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title='Model Weights in Ensemble',
                yaxis_title='Weight Magnitude',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"""
            **Interpretation:**
            - BERT has **{abs(bert_w)/abs(stylo_w):.1f}x** more influence than Stylometric
            - This is because BERT (73.9% acc) is more accurate than Stylometric (62.2% acc)
            - But Stylometric still adds complementary information about writing style!
            """)

            # Calculation breakdown
            st.markdown("#### Calculation Breakdown:")
            bert_contribution = results['bert']['score'] * bert_w
            stylo_contribution = results['stylometric']['score'] * stylo_w
            bias = results['ensemble']['weights']['bias']
            total = bert_contribution + stylo_contribution + bias

            st.code(f"""
BERT contribution:  {results['bert']['score']:.4f} × {bert_w:.3f} = {bert_contribution:.4f}
Stylo contribution: {results['stylometric']['score']:.4f} × {stylo_w:.3f} = {stylo_contribution:.4f}
Bias:               {bias:.4f}
                    ──────────
Sum:                {total:.4f}
After sigmoid:      {results['ensemble']['score']:.4f}
            """)

        with tab3:
            st.subheader("Model Score Comparison")

            # Score comparison
            scores_df = {
                'Model': ['BERT', 'Stylometric', 'Ensemble'],
                'Score': [
                    results['bert']['score'],
                    results['stylometric']['score'],
                    results['ensemble']['score']
                ]
            }

            fig = px.bar(
                scores_df,
                x='Model',
                y='Score',
                text='Score',
                title='Similarity Scores by Model',
                labels={'Score': 'Probability (Same Author)'}
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="Decision Threshold (0.5)")
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=400, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **How to read this:**
            - Scores > 0.5 predict "Same Author"
            - Scores < 0.5 predict "Different Authors"
            - Distance from 0.5 indicates confidence
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Built with Streamlit</strong> | 🎯 Ensemble: 73.9% accuracy, 0.823 AUC</p>
    <p style='font-size: 0.9em;'>Dataset: <a href='https://huggingface.co/datasets/swan07/authorship-verification'>swan07/authorship-verification</a> (325K pairs from 12 sources including PAN 2011-2020)</p>
    <p style='font-size: 0.8em;'>Manolache et al. (2021). <em>Transferring BERT-like Transformers' Knowledge for Authorship Verification</em>. <a href='https://arxiv.org/abs/2112.05125'>arXiv:2112.05125</a></p>
</div>
""", unsafe_allow_html=True)
