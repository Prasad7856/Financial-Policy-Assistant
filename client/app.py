import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatUI import render_chat

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Financial Policy Assistant",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

h1 {
    font-weight: 700;
}

hr {
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("🏦 Financial Policy Assistant")

st.caption(
    "AI-powered Retrieval-Augmented Generation (RAG) system for financial policy intelligence"
)

# st.divider()

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------
render_uploader()

st.divider()

# ---------------------------------------------------
# Chat Section
# ---------------------------------------------------
render_chat()

# ---------------------------------------------------
# Download History
# ---------------------------------------------------
render_history_download()