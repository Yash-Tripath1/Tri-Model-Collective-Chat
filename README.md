# Collective AI Chat
A simple local app to chat with Claude, Groq, and Gemini.
It can route prompts between models to save tokens.
Optional web search can add live grounding.
You can upload text files or PDFs for context — PDF text is extracted in the browser.
Turn on **Thinking mode** for deeper multi-step reasoning (extra reason step + verifier + higher token budgets).
The app shows token usage and estimated cost.
Run it with: `py app.py`
Then open: `http://localhost:8000`
Add your API keys in the sidebar and start chatting.
For PDFs: upload a file, click **Analyze PDF prompt**, then Send.
If one provider fails, try another model or key.