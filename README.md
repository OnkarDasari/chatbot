💬 Groq Chat Streamlit App

A sleek and efficient multi-model chat application powered by Groq and authenticated via Supabase with Google OAuth. Chat histories are stored per user and model for persistence and retrieval.
🔧 Features

    ✅ Google Login with Supabase OAuth (paste redirected URL to complete login)

    💾 User-specific chat history stored in Supabase

    🤖 Support for multiple Groq models:

        LLaMA 3.3 70B Versatile

        LLaMA 3.1 8B Instant

        Gemma 2 9B IT

        llama3-70b-8192

        llama3-8b-8192

    📊 Control Max Tokens and Temperature for responses

    ♻️ Clear chat history & logout options

    ⚡ Streaming responses with avatars

🚀 Live Demo

🌐 [Open the App](https://groqchatonkar.streamlit.app/)

🛠️ Setup Instructions
1. Clone the repository
`git clone https://github.com/yourusername/groq-chat-streamlit.git`
`cd groq-chat-streamlit`

2. Set up your environment
Use a virtual environment (recommended):
`python -m venv venv`
`source venv/bin/activate  # or .\venv\Scripts\activate on Windows`

Install the required packages:
`pip install -r requirements.txt`

3. Configure your secrets
Add a .streamlit/secrets.toml file with:
`SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-or-service-role-key"
GROQ_API_KEY = "your-groq-api-key"`

