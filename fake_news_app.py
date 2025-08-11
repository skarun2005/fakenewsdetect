# ------------------------------
# Fake News Detection & Source Credibility Scoring with BERT + GNN (Streamlit App)
# Enhanced Version with Pre-training, Advanced Features, and Improvements
# ------------------------------

# ------------------------------
# STEP 1: Install Required Packages
# ------------------------------
# Run in terminal:
# pip install torch torchvision torchaudio transformers streamlit scikit-learn pandas numpy networkx torch-geometric beautifulsoup4 requests matplotlib plotly feedparser

# Note: Added feedparser for RSS parsing in real-time search.

# ------------------------------
# STEP 2: Import Libraries
# ------------------------------
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import io
import os
import feedparser  # For real-time news search via RSS

# ------------------------------
# STEP 3: Define Models
# ------------------------------

class BERTEncoder(nn.Module):
    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ------------------------------
# STEP 4: Load Models with Pre-trained Support
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# For BERT Classification (used for fake news detection)
bert_classifier_path = 'bert_classifier.pth'
if os.path.exists(bert_classifier_path):
    bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    bert_classifier.load_state_dict(torch.load(bert_classifier_path))
    st.info("Loaded pre-trained BERT classifier.")
else:
    bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

bert_encoder = BERTEncoder().to(device)  # For embeddings in GNN

# For GNN
gnn_model_path = 'gnn_model.pth'
if os.path.exists(gnn_model_path):
    gnn_model = GNNModel(768, 256, 2).to(device)
    gnn_model.load_state_dict(torch.load(gnn_model_path))
    st.info("Loaded pre-trained GNN model.")
else:
    gnn_model = GNNModel(768, 256, 2).to(device)

# NER Pipeline for Advanced Source Extraction
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=0 if torch.cuda.is_available() else -1)

# ------------------------------
# STEP 5: Helper Functions
# ------------------------------

def get_bert_embedding(text):
    encoded = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    with torch.no_grad():
        embedding = bert_encoder(input_ids, attention_mask)
    return embedding.squeeze(0)

def detect_fake_news(text):
    encoded = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    with torch.no_grad():
        outputs = bert_classifier(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    score = torch.softmax(logits, dim=1)[0][1].item()  # Probability of being real
    label = "Fake" if score < 0.5 else "Real"
    return label, round(score, 3)

def build_source_graph(source_texts):
    embeddings = [get_bert_embedding(text).cpu().numpy() for text in source_texts]
    G = nx.Graph()
    for i, emb in enumerate(embeddings):
        G.add_node(i, x=emb)
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if sim > 0.9:
                G.add_edge(i, j)
    x = torch.tensor([G.nodes[i]['x'] for i in G.nodes], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

def source_credibility(source_texts):
    data = build_source_graph(source_texts)
    data = data.to(device)
    gnn_model.eval()
    with torch.no_grad():
        credibility_scores = gnn_model(data.x, data.edge_index)
    scores = torch.softmax(credibility_scores, dim=1)[:, 1].cpu().numpy()
    return scores

def scrape_article_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()[:2000]  # Limit to 2000 characters
    except Exception as e:
        return f"Error scraping URL: {e}"

def extract_sources_from_url(url, advanced=False):
    content = scrape_article_from_url(url)
    sources = []
    if content.startswith("Error"):
        return sources
    # Basic: Extract hyperlinks
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        sources = [link.get('href') for link in links if link.get('href') and 'http' in link.get('href')]
        sources = list(set(sources))[:5]  # Limit to 5 unique sources
    except:
        pass
    # Advanced: Use NER to extract organizations from text
    if advanced:
        entities = ner_pipeline(content)
        org_sources = [entity['word'] for entity in entities if entity['entity'].startswith('B-ORG') or entity['entity'].startswith('I-ORG')]
        sources.extend(list(set(org_sources)))  # Add unique organizations
    return sources[:10]  # Limit total

def search_real_time_news(query):
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:5]:  # Limit to 5
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'description': entry.description
        })
    return articles

# Simple GNN Training (Dummy example using random graph data)
def train_gnn_dummy():
    # Create dummy graph data
    x = torch.rand((10, 768), dtype=torch.float).to(device)
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=torch.long).to(device)
    y = torch.tensor([0,1,0,1,0,1,0,1,0,1], dtype=torch.long).to(device)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    gnn_model.train()
    for epoch in range(10):  # Small training
        optimizer.zero_grad()
        out = gnn_model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    torch.save(gnn_model.state_dict(), gnn_model_path)
    return "GNN trained on dummy data and saved."

# ------------------------------
# STEP 6: Fine-tune BERT on Dataset with Progress Bar
# ------------------------------
def fine_tune_bert(df, epochs=1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()  # Assume 0: Fake, 1: Real
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    inputs = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(inputs, attention_mask, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(bert_classifier.parameters(), lr=2e-5)
    
    progress_bar = st.progress(0)
    bert_classifier.train()
    total_steps = epochs * len(loader)
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = [b.to(device) for b in batch]
            outputs = bert_classifier(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            progress_bar.progress(step / total_steps)
    torch.save(bert_classifier.state_dict(), bert_classifier_path)
    return "BERT fine-tuned and saved."

# ------------------------------
# STEP 7: Create Default Dataset
# ------------------------------
def create_default_dataset():
    data = {
        'text': [
            "The government is hiding evidence of alien life on Mars.",
            "New study confirms coffee reduces risk of heart disease.",
            "Local election results show a surprising upset in the mayoral race.",
            "Celebrity caught in scandal with fake charity organization.",
            "Scientists discover new species of fish in Pacific Ocean."
        ],
        'label': [0, 1, 1, 0, 1]  # 0: Fake, 1: Real
    }
    return pd.DataFrame(data)

# Function to Download ISOT Dataset (Placeholder - User needs to handle Kaggle API or manual download)
def download_isot_dataset():
    st.info("To use a larger dataset like ISOT, download from https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets and upload via the uploader.")
    # For auto-download, user can set up Kaggle API, but omitted here for simplicity.

# ------------------------------
# STEP 8: Streamlit Web App
# ------------------------------
def main():
    # Set page configuration for a creative look
    st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="wide")

    # Custom CSS for improved UI
    st.markdown("""
        <style>
        .main { background-color: #f0f2f6; }
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; }
        .stTextInput, .stTextArea { background-color: #ffffff; border-radius: 10px; }
        .title { font-size: 2.5em; color: #1E3A8A; text-align: center; }
        .subheader { font-size: 1.5em; color: #3B82F6; }
        .sidebar .sidebar-content { background-color: #1E3A8A; color: white; }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown('<div class="title">🧠 Fake News Detection & Source Credibility Scoring</div>', unsafe_allow_html=True)
    st.markdown("""
        This app leverages **BERT** for fake news detection and **GNN** for source credibility scoring. 
        Upload a dataset, paste news content, or provide a URL to analyze articles and their sources.
        Enhanced with pre-trained model loading, advanced source extraction, real-time search, and UI improvements.
    """)

    # Sidebar for navigation
    st.sidebar.markdown('<div class="subheader">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.selectbox("Choose a feature:", ["Home", "Analyze News", "Real-Time Search", "Source Credibility", "Upload & Fine-Tune"])

    # Load default dataset
    default_dataset = create_default_dataset()

    if page == "Home":
        st.markdown('<div class="subheader">Welcome to the Fake News Detector</div>', unsafe_allow_html=True)
        st.write("Use the sidebar to navigate to different features. Start by analyzing news, scoring sources, or uploading a dataset to fine-tune the model.")
        download_isot_dataset()

    elif page == "Analyze News":
        st.markdown('<div class="subheader">🔍 Analyze a News Article</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            news_input = st.text_area("Enter news content manually:", height=200)
            url_input = st.text_input("OR enter a news article URL:")
        
        with col2:
            if st.button("Analyze Article"):
                if url_input and not news_input:
                    scraped = scrape_article_from_url(url_input)
                    st.write("**Scraped Content:**")
                    st.write(scraped[:500] + "..." if len(scraped) > 500 else scraped)
                    news_input = scraped
                if news_input:
                    label, score = detect_fake_news(news_input)
                    st.markdown(f"**Result:** {label} (Confidence: {score})")
                    # Visualize result with animation
                    fig = px.pie(values=[score, 1-score], names=['Real', 'Fake'], title='News Authenticity')
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig)

    elif page == "Real-Time Search":
        st.markdown('<div class="subheader">🔎 Real-Time News Search</div>', unsafe_allow_html=True)
        query = st.text_input("Enter search query for real-time news:")
        if st.button("Search News"):
            articles = search_real_time_news(query)
            if articles:
                for article in articles:
                    st.write(f"**Title:** {article['title']}")
                    st.write(f"**Description:** {article['description']}")
                    st.write(f"**Link:** {article['link']}")
                    if st.button(f"Analyze {article['title'][:20]}...", key=article['link']):
                        label, score = detect_fake_news(article['description'])
                        st.write(f"**Detection:** {label} (Confidence: {score})")
            else:
                st.warning("No articles found.")

    elif page == "Source Credibility":
        st.markdown('<div class="subheader">🌐 Source Credibility Scoring</div>', unsafe_allow_html=True)
        advanced_extraction = st.checkbox("Use Advanced NLP Source Extraction (NER for Organizations)")
        source_option = st.radio("Select source input method:", ["Manual Entry", "From URL", "From Dataset"])

        if source_option == "Manual Entry":
            source_texts_input = st.text_area("Enter sources (comma-separated):", height=100)
            if st.button("Score Sources"):
                source_texts = [s.strip() for s in source_texts_input.split(",") if s.strip()]
                if len(source_texts) >= 2:
                    scores = source_credibility(source_texts)
                    st.write("**Source Credibility Scores:**")
                    for src, scr in zip(source_texts, scores):
                        st.write(f"- {src[:40]}...: {round(scr, 3)}")
                    # Visualize scores with bar chart
                    fig = px.bar(x=source_texts, y=scores, labels={'x': 'Source', 'y': 'Credibility Score'}, title='Source Credibility')
                    st.plotly_chart(fig)
                else:
                    st.warning("Please provide at least two sources.")

        elif source_option == "From URL":
            url_input = st.text_input("Enter a URL to extract sources:")
            if st.button("Score Sources from URL"):
                sources = extract_sources_from_url(url_input, advanced=advanced_extraction)
                if sources:
                    st.write("**Extracted Sources:**")
                    for src in sources:
                        st.write(f"- {src}")
                    scores = source_credibility(sources)
                    st.write("**Source Credibility Scores:**")
                    for src, scr in zip(sources, scores):
                        st.write(f"- {src[:40]}...: {round(scr, 3)}")
                    # Visualize scores
                    fig = px.bar(x=sources, y=scores, labels={'x': 'Source', 'y': 'Credibility Score'}, title='Source Credibility')
                    st.plotly_chart(fig)
                else:
                    st.error("No valid sources found in the URL.")

        elif source_option == "From Dataset":
            st.write("Using default dataset sources:")
            source_texts = default_dataset['text'].tolist()
            if st.button("Score Sources from Dataset"):
                scores = source_credibility(source_texts)
                st.write("**Source Credibility Scores:**")
                for src, scr in zip(source_texts, scores):
                    st.write(f"- {src[:40]}...: {round(scr, 3)}")
                # Visualize scores
                fig = px.bar(x=source_texts, y=scores, labels={'x': 'Source', 'y': 'Credibility Score'}, title='Source Credibility')
                st.plotly_chart(fig)

    elif page == "Upload & Fine-Tune":
        st.markdown('<div class="subheader">🧠 Upload Dataset & Fine-Tune Models</div>', unsafe_allow_html=True)
        st.write("Default dataset preview:")
        st.write(default_dataset)
        
        uploaded_file = st.file_uploader("Upload a CSV dataset with 'text' and 'label' columns:", type="csv")
        epochs = st.slider("Number of Epochs:", 1, 5, 1)
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Dataset Preview:", df.head())
            if st.button("Fine-Tune BERT on Uploaded Dataset"):
                message = fine_tune_bert(df, epochs)
                st.success(message)
        else:
            if st.button("Fine-Tune BERT on Default Dataset"):
                message = fine_tune_bert(default_dataset, epochs)
                st.success(message)
        
        if st.button("Train GNN on Dummy Data"):
            message = train_gnn_dummy()
            st.success(message)

if __name__ == "__main__":
    main()
