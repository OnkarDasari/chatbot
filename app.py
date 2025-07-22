import streamlit as st
from typing import Generator
from groq import Groq
from supabase import create_client, Client
import urllib.parse
import uuid
import atexit
from datetime import datetime, timezone, timedelta

# Your Supabase keys (you can load from st.secrets too)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

redirect_uri = "http://localhost:8501/"

# Auth state
if "user" not in st.session_state:
    st.session_state.user = None
if "is_guest" not in st.session_state:
    st.session_state.is_guest = False

# Show login options if not logged in
if not st.session_state.user and not st.session_state.is_guest:
    st.title("🔐 Login to Groq Chat")

    col1, col2 = st.columns(2)
    with col1:
        oauth_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={urllib.parse.quote(redirect_uri)}"
        st.markdown(f"[🟢 Sign in with Google]({oauth_url})")

    with col2:
        if st.button("➡️ Continue as Guest"):
            guest_id = str(uuid.uuid4())
            st.session_state.user = {"id": guest_id, "email": "guest"}
            st.session_state.is_guest = True
            st.success("Using app as guest.")
            st.rerun()

    # After redirect
    auth_url = st.text_input("Paste the full redirected URL after signing in:")
    if auth_url:
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(auth_url)
        fragments = parse_qs(parsed.fragment)
        access_token = fragments.get("access_token", [None])[0]

        if access_token:
            user = supabase.auth.get_user(access_token)
            st.session_state.user = user.user
            st.session_state.is_guest = False
            st.success("Logged in as " + st.session_state.user.email)
            st.rerun()
        else:
            st.error("Could not extract access_token.")
    st.stop()

user_id = st.session_state.user["id"] if isinstance(st.session_state.user, dict) else st.session_state.user.id
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

supabase.table("chat_history") \
    .delete() \
    .lte("timestamp", cutoff.isoformat()) \
    .eq("is_guest", True) \
    .execute()


# --- Logout and Clear History Buttons ---
st.sidebar.title("Session Options")

# Log Out
if st.sidebar.button("🚪 Log Out"):
    if st.session_state.is_guest:
        supabase.table("chat_history") \
            .delete() \
            .eq("user_id", user_id) \
            .execute()
    st.session_state.user = None
    st.session_state.is_guest = False
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
def log_chat(user_id, message, response, model, chat_id=None, is_guest=False):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "message": message,
        "response": response,
        "model": model,
        "chat_id": chat_id,
        "is_guest": is_guest
    }).execute()



st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Chat App")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🧠")

st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

def fetch_chat_history(user_id: str, model: str, chat_id: int):
    response = supabase.table("chat_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("model", model) \
        .eq("chat_id", chat_id) \
        .order("timestamp", desc=False) \
        .execute()
    return response.data




if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Select a model"


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
    selected = st.selectbox(
        "Choose a model:",
        options=["Select a model"] + list(models.keys()),
        format_func=lambda x: models[x]["name"] if x in models else x
    )

if selected == "Select a model":
    st.warning("Please select a model to continue.")
    st.stop()
else:
    model_option = selected


# Only set chat_id if it hasn't already been set
if "chat_id" not in st.session_state or st.session_state.chat_id is None:
    # Fetch latest chat_id for this user and model
    existing_chats = supabase.table("chat_history").select("chat_id").match({
        "user_id": user_id,
        "model": model_option
    }).order("chat_id", desc=True).limit(1).execute()

    if existing_chats.data:
        st.session_state.chat_id = existing_chats.data[0]["chat_id"]


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
    st.session_state.chat_id = None  # Reset chat ID

st.sidebar.title("Chat History")

# ➕ New Chat button (must come BEFORE chat_id check!)
if st.sidebar.button("➕ New Chat"):
    # Get highest existing chat_id for this user+model
    result = supabase.table("chat_history") \
        .select("chat_id") \
        .eq("user_id", user_id) \
        .eq("model", model_option) \
        .order("chat_id", desc=True) \
        .limit(1) \
        .execute()

    previous_id = result.data[0]["chat_id"] if result.data else 0
    new_chat_id = previous_id + 1

    # Insert placeholder row
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "model": model_option,
        "chat_id": new_chat_id,
        "message": None,
        "response": None,
        "is_guest": st.session_state.is_guest  # ✅ This is required
    }).execute()

    st.session_state.chat_id = new_chat_id
    st.session_state.messages = []



# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.selected_model = model_option
    st.session_state.messages = []
    st.session_state.chat_id = None  # Reset chat ID

# 🧠 Auto-load previous chat if none selected yet
if st.session_state.chat_id is None:
    existing_chats = supabase.table("chat_history") \
        .select("chat_id") \
        .eq("user_id", user_id) \
        .eq("model", model_option) \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()

    if existing_chats.data:
        st.session_state.chat_id = existing_chats.data[0]["chat_id"]

# ✅ Ensure messages are loaded for current chat_id
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    if st.session_state.chat_id is not None:
        history = fetch_chat_history(user_id, model_option, st.session_state.chat_id)
        for row in history:
            if row["message"]:
                st.session_state.messages.append({"role": "user", "content": row["message"]})
            if row["response"]:
                st.session_state.messages.append({"role": "assistant", "content": row["response"]})



# 💬 Load all chats for selected model for sidebar display
def fetch_all_chats_for_model(user_id, model):
    result = supabase.table("chat_history") \
        .select("chat_id") \
        .eq("user_id", user_id) \
        .eq("model", model) \
        .order("chat_id") \
        .execute()

    chat_ids = sorted(set(row["chat_id"] for row in result.data))
    return chat_ids

chat_list = fetch_all_chats_for_model(user_id, st.session_state.selected_model)

st.sidebar.markdown("### 💬 Your Chats")
for cid in chat_list:
    is_active = cid == st.session_state.chat_id
    chat_label = f"👉 Chat {cid}" if is_active else f"Chat {cid}"
    if st.sidebar.button(chat_label, key=f"chat_{cid}"):
        st.session_state.chat_id = cid
        st.session_state.messages = []

        history = fetch_chat_history(user_id, st.session_state.selected_model, cid)
        for row in history:
            if row["message"]:
                st.session_state.messages.append({"role": "user", "content": row["message"]})
            if row["response"]:
                st.session_state.messages.append({"role": "assistant", "content": row["response"]})
        st.rerun()

# 🗑️ Delete currently selected chat
if st.session_state.chat_id is not None:
    if st.sidebar.button("🗑️ Delete Chat", use_container_width=True):
        supabase.table("chat_history") \
            .delete() \
            .eq("user_id", user_id) \
            .eq("model", model_option) \
            .eq("chat_id", st.session_state.chat_id) \
            .execute()

        st.session_state.chat_id = None
        st.session_state.messages = []
        st.rerun()



## Initialize chat history only if chat_id is set
if "messages" not in st.session_state:
    st.session_state.messages = []
    if st.session_state.chat_id is not None:
        history = fetch_chat_history(user_id, model_option, st.session_state.chat_id)
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

if st.session_state.chat_id is None:
    st.warning("Please start a new chat using the ➕ New Chat button.")
    st.stop()

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
        log_chat(user_id=user_id, message=prompt, response=full_response,
         model=model_option, chat_id=st.session_state.chat_id,
         is_guest=st.session_state.is_guest)

    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
        # Log chat to Supabase
        log_chat(user_id=user_id, message=prompt, response=combined_response,
         model=model_option, chat_id=st.session_state.chat_id,
         is_guest=st.session_state.is_guest)


if st.session_state.is_guest:
    def cleanup_guest_data():
        supabase.table("chat_history") \
            .delete() \
            .eq("user_id", user_id) \
            .execute()
    atexit.register(cleanup_guest_data)
