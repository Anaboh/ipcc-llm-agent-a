# 🌍 IPCC Climate Reports LLM Agent

AI-powered analysis of IPCC climate reports with multi-LLM support and FAISS integration.

## 🚀 Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python app.py`
5. Add environment variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `OPENAI_API_KEY`: (Optional) OpenAI API key
   - `ANTHROPIC_API_KEY`: (Optional) Anthropic API key
   - `GEMINI_API_KEY`: (Optional) Google Gemini API key

## 🔧 Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
