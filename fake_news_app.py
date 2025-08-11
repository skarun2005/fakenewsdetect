# Fake News Detection & Source Credibility Scoring with BERT + GNN (Streamlit App)

"""
Overview:
This application performs fake news detection using BERT (a powerful NLP model) and scores the credibility of the news sources using a Graph Neural Network (GNN). Additional features such as web scraping are planned to allow automatic collection and analysis of real-time news articles.
"""

# ------------------------------
# STEP 1: Install Required Packages
# ------------------------------
# pip install torch torchvision torchaudio transformers streamlit scikit-learn pandas numpy networkx torch-geometric beautifulsoup4 requests matplotlib plotly

# ------------------------------
# STEP 2: Import Libraries
# ------------------------------
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
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
import matplotlib.pyplot as plt
import plotly.express as px

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
# STEP 4: Load Models
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_encoder = BERTEncoder().to(device)
gnn_model = GNNModel(768, 256, 2).to(device)

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
    embedding = get_bert_embedding(text)
    score = torch.sigmoid(embedding.mean()).item()
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
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()[:2000]  # Limit to 2000 characters
    except Exception as e:
        return f"Error scraping URL: {e}"

# ------------------------------
# STEP 6: Fine-tune BERT on Dataset
# ------------------------------
def fine_tune_bert(df):
    from transformers import BertForSequenceClassification, AdamW
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    inputs = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(inputs, attention_mask, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(1):
        for batch in loader:
            batch = [b.to(device) for b in batch]
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

# ------------------------------
# STEP 7: Streamlit Web App
# ------------------------------
st.title("🧠 Fake News Detection & Source Credibility Scoring")
st.write("This app uses BERT for fake news detection and GNN for source credibility scoring. Optionally, you can scrape news content from URLs, upload datasets, and fine-tune BERT.")

# Default dataset loading (optional)
try:
    default_dataset = pd.read_csv("sample_news_dataset.csv")  # <--- You can change filename here
except Exception as e:
    default_dataset = None

st.subheader("🧠 Upload Dataset (CSV format)")
uploaded_file = st.file_uploader("Choose a dataset with 'text' and 'label' columns:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    if st.button("Fine-Tune BERT on Uploaded Dataset"):
        model = fine_tune_bert(df)
        st.success("Model fine-tuned successfully.")
else:
    if default_dataset is not None:
        st.write("Loaded default dataset: sample_news_dataset.csv")
        if st.button("Fine-Tune BERT on Default Dataset"):
            model = fine_tune_bert(default_dataset)
            st.success("Model fine-tuned successfully.")

st.subheader("🔍 Analyze a News Article")
news_input = st.text_area("Enter news content manually:")
url_input = st.text_input("OR enter a news article URL:")

if st.button("Analyze Article"):
    if url_input and not news_input:
        scraped = scrape_article_from_url(url_input)
        st.write("**Scraped Content:**")
        st.write(scraped)
        news_input = scraped
    if news_input:
        label, score = detect_fake_news(news_input)
        st.write(f"**Fake News Detection Result:** {label} (Confidence: {score})")

st.subheader("🌐 Source Credibility Scoring")
source_texts_input = st.text_area("Enter sources (comma-separated):")
if st.button("Score Sources"):
    source_texts = [s.strip() for s in source_texts_input.split(",") if s.strip()]
    if len(source_texts) >= 2:
        scores = source_credibility(source_texts)
        st.write("**Source Credibility Scores:**")
        for src, scr in zip(source_texts, scores):
            st.write(f"- {src[:40]}...: {round(scr, 3)}")
    else:
        st.warning("Please provide at least two sources.")


