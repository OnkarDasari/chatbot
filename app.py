import streamlit as st
from typing import Generator
from groq import Groq
from supabase import create_client, Client
import urllib.parse


# Your Supabase keys (you can load from st.secrets too)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

redirect_uri = "https://groqchatonkar.streamlit.app/"

# Auth state
if "user" not in st.session_state:
    st.session_state.user = None

# Show login if not logged in
if not st.session_state.user:
    st.title("Login")

    oauth_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={urllib.parse.quote(redirect_uri)}"
    st.markdown(f"[🟢 Sign in with Google]({oauth_url})")

    # After redirect, user will paste back the full URL
    auth_url = st.text_input("Paste the full redirected URL after signing in:")

    if auth_url:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(auth_url)
        fragments = parse_qs(parsed.fragment)

        access_token = fragments.get("access_token", [None])[0]

        if access_token:
            user = supabase.auth.get_user(access_token)
            st.session_state.user = user.user
            st.success("Logged in as " + st.session_state.user.email)
            st.rerun()
        else:
            st.error("Could not extract access_token.")
    st.stop()

user_id = st.session_state.user.id

# --- Logout and Clear History Buttons ---
st.sidebar.title("Session Options")

# Log Out
if st.sidebar.button("🚪 Log Out"):
    st.session_state.user = None
    st.rerun()

# Clear chat history for current user and model
if st.sidebar.button("🧹 Clear Chat History"):
    supabase.table("chat_history") \
        .delete() \
        .eq("user_id", user_id) \
        .eq("model", st.session_state.selected_model) \
        .execute()
    st.session_state.messages = []
    st.success("Chat history cleared.")
    st.rerun()

# Function to log chat messages to Supabase
def log_chat(user_id: str, message: str, response: str, model: str):
    data = {
        "user_id": user_id,
        "message": message,
        "response": response,
        "model": model,
    }
    print(st.session_state)
    supabase.table("chat_history").insert(data).execute()


st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Chat App")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("")

st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

def fetch_chat_history(user_id: str, model: str):
    response = supabase.table("chat_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("model", model) \
        .order("timestamp", desc=False) \
        .execute()
    return response.data




if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma2-9b-it": {"name": "Gemma2-9b-it", "tokens": 8192, "developer": "Google"},
    "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000, "developer": "Meta"},
    "llama-3.1-8b-instant" : {"name": "LLaMA3.1-8b-instant", "tokens": 128000, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
}

# Layout for model selection and max_tokens slider
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=1
    )

max_tokens_range = models[model_option]["tokens"]

with col2:
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,
        max_value=models[model_option]["tokens"],
        value=min(32768, models[model_option]["tokens"]),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response."
    )

with col3:
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Controls randomness. Lower is more deterministic; higher is more creative."
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.selected_model = model_option
    st.session_state.messages = []

    history = fetch_chat_history(user_id, model_option)
    for row in history:
        if row["message"]:
            st.session_state.messages.append({"role": "user", "content": row["message"]})
        if row["response"]:
            st.session_state.messages.append({"role": "assistant", "content": row["response"]})



# Initialize chat history and selected model
if "messages" not in st.session_state:
    history = fetch_chat_history(user_id, model_option)
    st.session_state.messages = []

    for row in history:
        if row["message"]:
            st.session_state.messages.append({"role": "user", "content": row["message"]})
        if row["response"]:
            st.session_state.messages.append({"role": "assistant", "content": row["response"]})

# ✅ Now we can render them
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
        # Log chat to Supabase
        log_chat(user_id=user_id, message=prompt, response=full_response, model=model_option)
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
        # Log chat to Supabase
        log_chat(user_id=user_id, message=prompt, response=combined_response, model=model_option)