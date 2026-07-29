"""RAG Search Frontend — Reference Script.

Reference script. See `rag_api.py` for the backend this calls over HTTP.

Methods covered:
- Calling a RAG backend over HTTP from a Streamlit app (`requests.post`)
- Regex-extracting citation markers (`[0]`, `[Document 0]`) from a generated answer
- Mapping cited IDs back to the source chunks returned alongside the answer
- Showing each cited source in an `st.expander` with an `st.download_button` for the original file

Use this as a reference when: you need a thin UI over an existing RAG API that surfaces which source
documents backed each answer.

Don't use this as a reference for: the retrieval/generation logic itself (see `rag_api.py`) — this
file only renders what the API already computed.

Requires (not part of the repo's shared pyproject.toml — install separately): streamlit, requests.

Run: streamlit run rag_streamlit_app.py
Requires rag_api.py running and reachable at API_URL (default http://127.0.0.1:8000/query).
"""

import json
import os
import re

import requests
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/query")

st.set_page_config(page_title="RAG Document Search", page_icon=":mag:", layout="centered")
st.title("Document Search with RAG")

question = st.text_input("Ask a question about the indexed documents:", "")

if st.button("Ask"):
    st.write(f'Question: "{question}"')

    response = requests.post(
        API_URL,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps({"query": question}),
    )
    payload = response.json()
    answer = payload["answer"]
    sources = payload["sources"]

    st.markdown(answer)

    # The system prompt in rag_api.py instructs the model to cite sources as "[0]", "[1]", etc. --
    # this regex pulls those bracketed ids back out so the matching source chunks can be shown below.
    citation_pattern = re.compile(r"\[Document\s[0-9]+\]|\[[0-9]+\]")
    cited_ids = {
        int(n)
        for marker in citation_pattern.findall(answer)
        for n in re.findall(r"\b\d+\b", marker)
    }
    cited_sources = [source for source in sources if source["id"] in cited_ids]

    for source in cited_sources:
        with st.expander(f"{source['id']} - {source['path']}"):
            st.write(source["content"])
            with open(source["path"], "rb") as f:
                st.download_button(
                    "Download source file",
                    f,
                    file_name=source["path"].split("/")[-1],
                    # One widget per cited source needs a distinct key, or Streamlit collapses them
                    # into a single stateful button.
                    key=f"download-{source['id']}",
                )
