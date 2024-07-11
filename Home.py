import streamlit as st
import pandas as pd
import altair as alt
import requests
import tiktoken
import json
from enum import Enum
from pathlib import Path

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import Document
from llama_index.core.schema import MetadataMode


# Configurations -----------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Visualize LlamaIndex Parsers", page_icon="🟦", initial_sidebar_state="collapsed", menu_items=None, layout='wide')
st.title("Visualize LlamaIndex text parsers 📄🦙📝")

if "FILE_NAMES" not in st.session_state.keys():
    st.session_state.FILE_NAMES = None


# Functions ----------------------------------------------------------------------------------------------------------------------------------------------------
def _try_to_get_github_pat_secret():
    if "GITHUB_PAT_TOKEN" not in st.secrets.keys():
        return None
    return st.secrets["GITHUB_PAT_TOKEN"]


def get_commits(file_path, owner, repo_name, branch=None, access_token=_try_to_get_github_pat_secret()):
    url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"  # https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#list-commits
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


class NodeParsers(Enum):
    # https://github.com/run-llama/llama_index/blob/589b3054e22535fc7a43baa2bbd52aacc439b4f7/llama-index-core/llama_index/core/node_parser/__init__.py
    MarkdownNodeParser = "MarkdownNodeParser()"
    SentenceSplitter = "SentenceSplitter()"
    TokenTextSplitter = "TokenTextSplitter()"

    @staticmethod
    def available_node_parsers():
        return list(map(lambda c: c.value, NodeParsers))

    @staticmethod
    def get_node_parser_params_dict(parser_name):
        params_dict = {}
        if parser_name == NodeParsers.SentenceSplitter.value:
            params_dict["chunk_size"] = SentenceSplitter().chunk_size
            params_dict["chunk_overlap"] = SentenceSplitter().chunk_overlap
        if parser_name == NodeParsers.TokenTextSplitter.value:
            params_dict["chunk_size"] = TokenTextSplitter().chunk_size
            params_dict["chunk_overlap"] = TokenTextSplitter().chunk_overlap
        return params_dict

    @staticmethod
    def get_node_parser(parser_name, params=None):
        if params is None:
            params = {}
        if parser_name == NodeParsers.MarkdownNodeParser.value:
            return MarkdownNodeParser()
        elif parser_name == NodeParsers.SentenceSplitter.value:
            return SentenceSplitter(**params)
        elif parser_name == NodeParsers.TokenTextSplitter.value:
            return TokenTextSplitter(**params)
        else:
            return None


def get_node_info(text_node, tokenizer=tiktoken.encoding_for_model("gpt-35-turbo"), emoji_value=100, emoji_icon=None):
    text = text_node.get_text()
    text_embed_mode = text_node.get_content(metadata_mode=MetadataMode.EMBED)
    text_llm_mode = text_node.get_content(metadata_mode=MetadataMode.LLM)

    info_dict = {"text": text,
                 "text_embed_mode": text_embed_mode,
                 "text_llm_mode": text_llm_mode,
                 "token_text_cnt": len(tokenizer.encode(text)),
                 "token_text_embed_cnt": len(tokenizer.encode(text_embed_mode)),
                 "token_text_llm_cnt": len(tokenizer.encode(text_llm_mode)),
                 "emoji_viz": None,
                 "emoji_value": emoji_value}

    if emoji_icon:
        emoji_viz = emoji_icon * int(info_dict['token_text_cnt'] / emoji_value)
        info_dict["emoji_viz"] = emoji_viz

    return info_dict


# App Interface ------------------------------------------------------------------------------------------------------------------------------------------------
col1a, col2a = st.columns(2)
# 1A) get Markdown files from [Setup files from GitHub]
with col1a:
    options_markdown_file = [None]
    if st.session_state.FILE_NAMES:
        options_markdown_file.extend(st.session_state.FILE_NAMES)
    selected_markdown = st.selectbox("**Select a file:** ", options_markdown_file, index=0)
# 2A) get GitHub commits for selected Markdown file
with col2a:
    options_commits, selected_pdf = [None], None
    if selected_markdown:
        selected_pdf = (Path(selected_markdown).with_suffix('.pdf'))
        returned_commits = get_commits(file_path=selected_markdown,
                                       owner=st.session_state.GITHUB_OWNER,
                                       repo_name=st.session_state.GITHUB_REPO,
                                       branch=st.session_state.GITHUB_BRANCH)
        options_commits.extend(returned_commits)
    selected_commit = st.selectbox("**Select the commit:** ", options_commits, index=0, help="Ordered by most recent commit. None = latest commit")


# Get Markdown text based on 1A & 2A selections
m_text = ""
if selected_markdown:
    m_text, _ = get_markdown_text_from_github(file_path=selected_markdown,
                                              owner=st.session_state.GITHUB_OWNER,
                                              repo_name=st.session_state.GITHUB_REPO,
                                              branch=st.session_state.GITHUB_BRANCH,
                                              commit=selected_commit)


col1b, col2b = st.columns(2)
height_b, ext_2b = 1000, 30
# 1B) display Markdown text & pdf document if available
with col1b:
    file_url, pdf_url = None, None
    if selected_markdown:
        use_commit_or_branch = selected_commit
        if use_commit_or_branch is None:
            use_commit_or_branch = st.session_state.GITHUB_BRANCH
        file_url = f'https://github.com/{st.session_state.GITHUB_OWNER}/{st.session_state.GITHUB_REPO}/blob/{use_commit_or_branch}/{selected_markdown}'
        pdf_url = f'https://github.com/{st.session_state.GITHUB_OWNER}/{st.session_state.GITHUB_REPO}/blob/{st.session_state.GITHUB_BRANCH}/{selected_pdf}'
    label_1b = f"**File text url:** [{selected_markdown}]({file_url}) **& if uploaded PDF:** [{selected_pdf}]({pdf_url})"
    selected_text = st.text_area(label=label_1b, value=m_text, height=height_b, placeholder="Paste text here...")
# 2B) display rendered Markdown text
with col2b:
    with st.container(height=height_b+ext_2b):
        st.markdown(selected_text)


col1c, col2c = st.columns(2)
# 1C) select node parser
with col1c:
    selected_parser_name = st.selectbox("**Select a node parser:** ", NodeParsers.available_node_parsers(), index=0)
    default_params_dict = NodeParsers.get_node_parser_params_dict(parser_name=selected_parser_name)
# 2C) add additional parameters
with col2c:
    selected_params_dict = None
    if len(default_params_dict) != 0:
        selected_params_dict = st.text_area("**Add additional parameters:**", value=json.dumps(default_params_dict))
        selected_params_dict = json.loads(selected_params_dict)


# Get node parser based on 1C & 2C selections
node_parser = NodeParsers.get_node_parser(parser_name=selected_parser_name, params=selected_params_dict)
# st.write(node_parser.__dict__)


# D) get node statistics and chunks based on 1B) & C) selections
EMOJI_ICON, EMOJI_VALUE = "🟦", 100
if st.checkbox("**Parse Documents**", help=f"{EMOJI_ICON} = {EMOJI_VALUE} tokens but please note int() rounding occurs"):
    # configure and run
    doc = Document(text=selected_text,
                   metadata={})  # https://github.com/run-llama/llama_index/blob/f599f1511488737066af8d33f4bfdb18907706c8/llama-index-core/llama_index/core/readers/file/base.py#L565C19-L565C63
    nodes = node_parser.get_nodes_from_documents([doc], show_progress=True)
    node_info = [get_node_info(n, emoji_value=EMOJI_VALUE, emoji_icon=EMOJI_ICON) for n in nodes]

    # display token statistics
    stats_tokens = [n.get('token_text_cnt') for n in node_info]
    stats_tokens_w_meta = [n.get('token_text_embed_cnt') for n in node_info]

    data_dict = {"node_i": list(range(len(stats_tokens))),
                 "token_cnt": stats_tokens}
    data_df = pd.DataFrame(data_dict)

    bar_chart = alt.Chart(data_df).mark_bar().encode(
        x=alt.X("node_i:N", title="Node"),
        y=alt.Y("token_cnt:Q", title="Token Count")
    )
    st.code(f"Token Statistics: {sum(stats_tokens)} tokens | {sum(stats_tokens_w_meta)} tokens w/ metadata")
    st.altair_chart(bar_chart, use_container_width=True)

    # display chunks as text and rendered markdown
    for i, ns in enumerate(node_info):
        st.code(f"""Node {i} : {ns['emoji_viz']} | {ns['token_text_cnt']} tokens | {ns['token_text_embed_cnt']} tokens w/ metadata""")
        col1d, col2d = st.columns(2)
        with col1d:
            st.text(ns["text"])
        with col2d:
            st.markdown(ns["text"])

    # E) Run Embedding
    cost_per_token_openai_ada = 0.0001 / 1000  # $0.0001 per 1000 tokens (https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
    usd_to_nok = 10.5
    st.code(f"Estimated cost to run: {sum(stats_tokens_w_meta) * cost_per_token_openai_ada * usd_to_nok:.2f} nok")

    if st.button("Run Embedding Stats"):
        st.write("TO DO")
