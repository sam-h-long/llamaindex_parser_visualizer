import streamlit as st
import requests
import json
from pathlib import Path

st.title("Setup to load files from Github:")

github_initial_inputs = '''{
        "GITHUB_OWNER": "sam-h-long",
        "GITHUB_REPO": "llamaparse_pdf_to_markdown",
        "GITHUB_BRANCH": "main",
        "FILE_TYPES": [".md", ".txt"],
        "IGNORE_FILES": ["README.md", "requirements.txt"]
    }'''

if "GITHUB_INPUTS" not in st.session_state.keys():
    st.session_state.GITHUB_INPUTS = github_initial_inputs
    st.session_state.GITHUB_OWNER = None
    st.session_state.GITHUB_REPO = None
    st.session_state.GITHUB_BRANCH = None


def _try_to_get_github_pat_secret():
    if "GITHUB_PAT_TOKEN" not in st.secrets.keys():
        return None
    return st.secrets["GITHUB_PAT_TOKEN"]


def get_file_names_from_github(json_, access_token=_try_to_get_github_pat_secret()):
    url = f'https://api.github.com/repos/{json_["GITHUB_OWNER"]}/{json_["GITHUB_REPO"]}/git/trees/{json_["GITHUB_BRANCH"]}?recursive=1'
    headers = {}
    if access_token:
        headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    file_names = []
    if response.status_code == 200:
        response_json = response.json()
        for unit in response_json["tree"]:
            # check is file not subdirectory
            if unit["type"] == "blob":
                unit_path = Path(unit["path"])
                # check if file is 1) correct file type 2) not in ignore list
                if unit_path.suffix in json_["FILE_TYPES"] and unit_path.name not in json_["IGNORE_FILES"]:
                    file_names.append(str(unit_path))
        return file_names, True
    else:
        st.write(f"Error getting file names from GitHub. Status code: {response.status_code}")
        return None, False


col1a, col2a = st.columns(2)
with col1a:
    edit_input = st.text_area(label="Edit configuration:", value=st.session_state.GITHUB_INPUTS, height=220)
    st.session_state.GITHUB_INPUTS = edit_input
    st.caption(
        "If repo is private, you need to create a [GitHub PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) and define it in [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management). Example: GITHUB_PAT_TOKEN = 'ghp_....XXXXXXXXXXXX'")
with col2a:
    st.write("Current configuration:")
    try:
        json_input = json.loads(st.session_state.GITHUB_INPUTS)
        st.write(json_input)
    except json.JSONDecodeError:
        st.write("Invalid JSON")

files, success = get_file_names_from_github(json_=json_input)
st.write(files)
if success:
    st.write("Files:")
    # these session variables will lag behind the current values until success occurs
    st.session_state.GITHUB_OWNER = json_input["GITHUB_OWNER"]
    st.session_state.GITHUB_REPO = json_input["GITHUB_REPO"]
    st.session_state.GITHUB_BRANCH = json_input["GITHUB_BRANCH"]
    # these session variables get initialized in Home.py
    st.session_state.GITHUB_BASEURL = f'https://github.com/{json_input["GITHUB_OWNER"]}/{json_input["GITHUB_REPO"]}/blob/{json_input["GITHUB_BRANCH"]}'
    st.session_state.FILE_NAMES = files
else:
    st.session_state.FILE_NAMES = None
    st.session_state.GITHUB_BASEURL = None

if st.button("Show session state"):
    st.write(st.session_state)
