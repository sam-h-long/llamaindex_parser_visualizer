import streamlit as st
import pandas as pd
import altair as alt
import requests
import tiktoken
from enum import Enum
from pathlib import Path

from llama_index.core.node_parser import MarkdownNodeParser, SimpleNodeParser
from llama_index.core.schema import Document
from llama_index.core.schema import MetadataMode

st.set_page_config(page_title="Visualize LlamaIndex Parsers", page_icon="🟦", initial_sidebar_state="collapsed", menu_items=None, layout='wide')
st.title("PDF to Markdown: visualize LlamaIndex text parsers 📄🦙📝")

if "GITHUB_BASEURL" not in st.session_state.keys():
    st.session_state.GITHUB_BASEURL = None

if "FILE_NAMES" not in st.session_state.keys():
    st.session_state.FILE_NAMES = None


def _try_to_get_github_pat_secret():
    if "GITHUB_PAT_TOKEN" not in st.secrets.keys():
        return None
    return st.secrets["GITHUB_PAT_TOKEN"]


def get_commits(file_path, owner, repo_name, branch=None, access_token=_try_to_get_github_pat_secret()):
    url = f"https://api.github.com/repos/{owner}/{repo_name}/commits" # https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#list-commits
    params = {"path": file_path}
    if branch:
        params["sha"] = branch
    headers = {}
    if access_token:
        headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, params=params, headers=headers)
    commit_ids = []
    if response.status_code == 200:
        commits = response.json()
        for commit in commits:
            commit_url = commit["commit"].get("url")
            commit_ids.append(commit_url.split("/")[-1])
        return commit_ids
    else:
        ValueError(f"Error fetching commits. Status code: {response.status_code}")
        return None


@st.cache_resource(show_spinner=True)
def get_markdown_text_from_github(file_path, owner, repo_name, branch, commit=None, override_url=None, token=_try_to_get_github_pat_secret()):
    raw_base_url = f"raw.githubusercontent.com/{owner}/{repo_name}"
    if commit:
        url = f"https://x-access-token:{token}@{raw_base_url}/{commit}/{file_path}"
    else:
        url = f"https://x-access-token:{token}@{raw_base_url}/{branch}/{file_path}"
    if override_url:
        url = override_url
    response = requests.get(url)
    if response.status_code == 200:
        return response.text, url
    else:
        ValueError(f"Failed to get markdown text from {url}")
        return None


class NodeParserConstants(Enum):
    # https://github.com/run-llama/llama_index/blob/589b3054e22535fc7a43baa2bbd52aacc439b4f7/llama-index-core/llama_index/core/node_parser/__init__.py
    MarkdownNodeParser = "MarkdownNodeParser"
    SentenceSplitter = "SentenceSplitter"

    @staticmethod
    def available_node_parsers():
        return list(map(lambda c: c.value, NodeParserConstants))


def get_node_info(text_node, tokenizer=tiktoken.encoding_for_model("gpt-35-turbo")):
    text = text_node.get_text()
    text_embed_mode = text_node.get_content(metadata_mode=MetadataMode.EMBED)
    text_llm_mode = text_node.get_content(metadata_mode=MetadataMode.LLM)

    info_dict = {"text": text,
                 "text_embed_mode": text_embed_mode,
                 "text_llm_mode": text_llm_mode,
                 "token_text_cnt": len(tokenizer.encode(text)),
                 "token_text_embed_cnt": len(tokenizer.encode(text_embed_mode)),
                 "token_text_llm_cnt": len(tokenizer.encode(text_llm_mode))}
    return info_dict


# A) Selections
col1a, col2a = st.columns(2)
with col1a:
    options_markdown_file = [None]
    if st.session_state.FILE_NAMES:
        options_markdown_file.extend(st.session_state.FILE_NAMES)
    selected_markdown = st.selectbox("Select a markdown file: ", options_markdown_file, index=0)
with col2a:
    options_commits, selected_pdf = [None], None
    if selected_markdown:
        selected_pdf = (Path(selected_markdown).with_suffix('.pdf'))
        returned_commits = get_commits(file_path=selected_markdown,
                                       owner=st.session_state.GITHUB_OWNER,
                                       repo_name=st.session_state.GITHUB_REPO,
                                       branch=st.session_state.GITHUB_BRANCH)
        options_commits.extend(returned_commits)
    selected_commit = st.selectbox("Select the commit: ", options_commits, index=0)


# Get Markdown text based on selections
m_text = ""
if selected_markdown:
    m_text, _ = get_markdown_text_from_github(file_path=selected_markdown,
                                              owner=st.session_state.GITHUB_OWNER,
                                              repo_name=st.session_state.GITHUB_REPO,
                                              branch=st.session_state.GITHUB_BRANCH,
                                              commit=selected_commit)

# B) Display Markdown text
col1b, col2b = st.columns(2)
height_b = 1000
with col1b:
    st.caption(f'Original .pdf document: [{selected_pdf}]({st.session_state.GITHUB_BASEURL}/{selected_pdf})')
    md = st.text_area(label=f"Markdown text is from: [{selected_markdown}]({st.session_state.GITHUB_BASEURL}/{selected_markdown})",
                      value=m_text, height=height_b, placeholder="Paste text here...")
with col2b:
    with st.container(height=height_b):
        st.markdown(md)

selected_node_parser = st.selectbox("Select a node parser: ", NodeParserConstants.available_node_parsers(), index=0, placeholder="Select an index...", )

# C) When clicked, run MarkdownNodeParser()
stats_tokens, stats_tokens_w_meta = list(), list()
emoji_icon, emoji_value = "🟦", 100
cost_per_token_openai_ada = 0.0001 / 1000  # $0.0001 per 1000 tokens (https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
usd_to_nok = 10.5
if st.button("Parse Documents", help=f"{emoji_icon} = {emoji_value} tokens but please note int() rounding occurs"):
    # https://github.com/run-llama/llama_index/blob/f599f1511488737066af8d33f4bfdb18907706c8/llama-index-core/llama_index/core/readers/file/base.py#L565C19-L565C63
    doc = Document(text=md, metadata={})
    nodes = MarkdownNodeParser().get_nodes_from_documents([doc], show_progress=True)
    for i, n in enumerate(nodes):
        info_i = get_node_info(n)
        emoji_viz_i = emoji_icon * int(info_i['token_text_cnt'] / emoji_value)
        st.code(f"""Node {i} : {emoji_viz_i} | {info_i['token_text_cnt']} tokens | {info_i['token_text_embed_cnt']} tokens w/ metadata""")
        stats_tokens.append(info_i['token_text_cnt'])
        stats_tokens_w_meta.append(info_i['token_text_embed_cnt'])
        col1c, col2c = st.columns(2)
        with col1c:
            st.text(info_i["text"])
            # st.text(info_i["text_embed_mode"])
        with col2c:
            st.markdown(info_i["text"])

    # D) Display token statistics
    data_dict = {"node_i": list(range(len(stats_tokens))),
                 "token_cnt": stats_tokens}
    data_df = pd.DataFrame(data_dict)

    # Create a bar chart using Altair
    bar_chart = alt.Chart(data_df).mark_bar().encode(
        x=alt.X("node_i:N", title="Node"),
        y=alt.Y("token_cnt:Q", title="Token Count")
    )

    st.code(f"Token Statistics: {sum(stats_tokens)} tokens | {sum(stats_tokens_w_meta)} tokens w/ metadata")
    st.altair_chart(bar_chart, use_container_width=True)

    # E) Run Embedding
    st.code(f"Estimated cost to run: {sum(stats_tokens_w_meta) * cost_per_token_openai_ada * usd_to_nok:.2f} nok")
    if st.button("Run Embedding Stats"):
        st.write("TO DO: Run Embedding Stats")
