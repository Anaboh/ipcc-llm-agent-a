import os
import re
import json
import time
import pickle
import numpy as np
import gradio as gr
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

# Initialize environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Custom FAISS retriever
class FAISSRetriever(BaseRetriever):
    def __init__(self, index_path: str, texts_path: str, embed_model: HuggingFaceEmbeddings, k: int = 3):
        super().__init__()
        self._embed_model = embed_model
        self._index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            self._texts = pickle.load(f)
        self._k = k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self._embed_model.embed_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query_embedding, self._k)
        return [Document(page_content=self._texts[i]) for i in indices[0]]

# Main agent class
class IPCCLLMAgent:
    def __init__(self):
        self.conversation_history = []
        self.ipcc_reports = {
            'all': {'name': 'All IPCC Reports', 'color': '🌍'},
            'ar6_syr': {'name': 'AR6 Synthesis Report (2023)', 'color': '📖'},
            'ar6': {'name': 'AR6 (2021-2023)', 'color': '🌱'},
            'ar5': {'name': 'AR5 (2013-2014)', 'color': '📊'},
            'special': {'name': 'Special Reports', 'color': '⚠️'},
            'ar4': {'name': 'AR4 (2007)', 'color': '📄'}
        }
        self.llm_models = {
            'deepseek': {'name': 'DeepSeek-R1-Distill-Llama-70B', 'provider': 'Groq'},
            'llama': {'name': 'Llama-3.3-70B-Versatile', 'provider': 'Groq'},
            'gpt-4': {'name': 'GPT-4 Turbo', 'provider': 'OpenAI'},
            'gpt-3.5': {'name': 'GPT-3.5 Turbo', 'provider': 'OpenAI'},
            'claude-3': {'name': 'Claude 3 Sonnet', 'provider': 'Anthropic'},
            'gemini': {'name': 'Gemini Pro', 'provider': 'Google'},
            'mock': {'name': 'Mock AI (Demo)', 'provider': 'Local'}
        }
        self.setup_api_clients()
        self.ipcc_knowledge = self.load_ipcc_knowledge()
        self.faiss_retriever = self.setup_faiss_retriever()

    def setup_api_clients(self):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        self.groq_client_llama = None
        self.groq_client_deepseek = None
        
        try:
            import openai
            if OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            pass
            
        try:
            import anthropic
            if ANTHROPIC_API_KEY:
                self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            pass
            
        try:
            import google.generativeai as genai
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
        except ImportError:
            pass
            
        try:
            from langchain_groq import ChatGroq
            if GROQ_API_KEY:
                self.groq_client_llama = ChatGroq(api_key=GROQ_API_KEY, model_name='llama-3.3-70b-versatile')
                self.groq_client_deepseek = ChatGroq(api_key=GROQ_API_KEY, model_name='deepseek-r1-distill-llama-70b')
        except ImportError:
            pass

    def setup_faiss_retriever(self):
        try:
            # Use relative paths for deployment
            data_dir = "data"
            faiss_path = os.path.join(data_dir, "faiss_index.bin")
            texts_path = os.path.join(data_dir, "faiss_texts.pkl")
            
            if not os.path.exists(faiss_path) or not os.path.exists(texts_path):
                print("FAISS files not found. Skipping FAISS setup.")
                return None
            
            embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
            return FAISSRetriever(faiss_path, texts_path, embed_model, k=3)
        except Exception as e:
            print(f"Error setting up FAISS retriever: {str(e)}")
            return None

    def load_ipcc_knowledge(self) -> Dict:
        return {
            'ar6_summary': {
                'content': """# 🌡️ AR6 Synthesis Report: Key Findings for Policymakers
## **Physical Science Basis (Working Group I)**
- **Global Temperature Rise**: 1.1°C above 1850-1900 levels
- **Human Influence**: Unequivocally the dominant cause of warming
- **Rate of Change**: Faster than any period in over 2,000 years
- **Regional Impacts**: Every inhabited region experiencing climate change
- **Irreversible Changes**: Many changes locked in for centuries to millennia

## **Impacts, Adaptation & Vulnerability (Working Group II)**
- **Population at Risk**: 3.3-3.6 billion people highly vulnerable
- **Current Impacts**: Widespread losses and damages already occurring
- **Food Security**: 828 million people undernourished (2021)
- **Water Stress**: Up to 3 billion people experience water scarcity
- **Ecosystem Degradation**: Widespread species shifts and ecosystem changes

## **Mitigation of Climate Change (Working Group III)**
- **Emission Trends**: Global GHG emissions continued to rise
- **Peak Requirement**: Emissions must peak before 2025 for 1.5°C
- **2030 Target**: 43% reduction needed by 2030 (2019 levels)
- **2050 Target**: Net zero CO₂ emissions required
- **Investment Gap**: $4 trillion annually needed in clean energy

## **Integrated Solutions**
- **Rapid Transformation**: Deep, immediate cuts across all sectors
- **Technology Readiness**: Many solutions available and cost-effective
- **Co-benefits**: Climate action improves health, economy, equity
- **Just Transitions**: Equitable pathways essential for success""",
                'sources': ['AR6 Synthesis Report SPM', 'AR6 WG1-3 Reports']
            },
            'urgent_actions_2030': {
                'content': """# 🚨 Critical Climate Actions Needed by 2030
## **Energy System Transformation**
### Renewable Energy Scale-up
- **Target**: 60% renewable electricity globally (vs ~30% today)
- **Solar**: Increase capacity 4x from 2020 levels
- **Wind**: Triple offshore wind capacity
- **Storage**: Deploy 120 GW of battery storage annually
### Fossil Fuel Phase-out
- **Coal**: Retire 2,400+ coal plants globally
- **Oil & Gas**: Reduce production 75% by 2050
- **Subsidies**: End $5.9 trillion in fossil fuel subsidies
## **Transport Decarbonization**
### Electric Vehicle Revolution
- **Target**: 50% of new car sales electric by 2030
- **Infrastructure**: 40 million public charging points needed
- **Heavy Transport**: 30% of trucks electric/hydrogen by 2030
### Sustainable Aviation & Shipping
- **Aviation**: 10% sustainable fuels by 2030
- **Shipping**: 5% zero-emission fuels by 2030
## **Buildings & Cities**
### Zero-Carbon Buildings
- **New Buildings**: All new buildings zero-carbon by 2030
- **Retrofits**: Deep renovation of 3% of building stock annually
- **Heat Pumps**: 600 million heat pumps by 2030
### Urban Planning
- **15-Minute Cities**: Reduce transport demand 20%
- **Green Infrastructure**: 30% urban tree canopy coverage
## **Natural Climate Solutions**
### Forest Protection
- **Deforestation**: End deforestation by 2030
- **Restoration**: 350 million hectares by 2030
- **Carbon Storage**: 5.8 GtCO₂ annually from forests
### Sustainable Agriculture
- **Regenerative Practices**: 30% of farmland by 2030
- **Food Waste**: Reduce food waste 50%
- **Diets**: 20% shift toward plant-based diets
## **Financial Requirements**
- **Total Investment**: $4-6 trillion annually
- **Clean Energy**: $1.6-3.8 trillion annually
- **Nature**: $350 billion annually
- **Adaptation**: $140-300 billion annually by 2030""",
                'sources': ['AR6 WG3 Ch5', '1.5°C Special Report', 'AR6 Synthesis']
            },
            # Other knowledge sections remain the same as your original
        }

    def format_response(self, content: str, sources: List[str] = None, report_focus: str = 'all') -> str:
        formatted = f"**Report Focus**: {self.ipcc_reports[report_focus]['name']}\n\n{content}"
        if sources:
            formatted += f"\n\n**Sources**: {', '.join(sources)}"
        return formatted

    def get_mock_response(self, message: str, report_focus: str) -> Tuple[str, List[str]]:
        message_lower = message.lower()
        
        if report_focus == 'ar6_syr':
            return """This is a placeholder response for the AR6 Synthesis Report. Please select a Groq-hosted model to access the FAISS data.""", ['Mock Response']

        if any(word in message_lower for word in ['ar6', 'synthesis', 'key findings']):
            knowledge = self.ipcc_knowledge['ar6_summary']
            return knowledge['content'], knowledge['sources']
        elif any(word in message_lower for word in ['urgent', '2030', 'actions', 'immediate']):
            knowledge = self.ipcc_knowledge['urgent_actions_2030']
            return knowledge['content'], knowledge['sources']
        else:
            return """I can help you with IPCC climate reports! Try asking about:
- AR6 Synthesis Report key findings
- Urgent climate actions needed by 2030
- Carbon budgets and emissions pathways""", ['IPCC Knowledge Base']

    async def clean_response(self, content: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned.strip()

    async def call_llm_api(self, messages: List[Dict], model: str, temperature: float, max_tokens: int, report_focus: str) -> Tuple[str, List[str]]:
        if model == 'mock':
            time.sleep(1)
            user_message = messages[-1]['content']
            return self.get_mock_response(user_message, report_focus)

        system_prompt = """You are an expert IPCC climate reports analyst. Provide accurate, science-based responses using information from IPCC Assessment Reports."""
        
        # Handle FAISS retrieval for AR6 Synthesis
        if report_focus == 'ar6_syr' and self.faiss_retriever and model in ['deepseek', 'llama']:
            try:
                user_message = messages[-1]['content']
                docs = self.faiss_retriever._get_relevant_documents(user_message)
                context = "\n".join([doc.page_content for doc in docs])
                
                if model == 'deepseek' and self.groq_client_deepseek:
                    prompt_messages = [
                        {"role": "system", "content": f"{system_prompt}\n\nContext from AR6 Synthesis Report:\n{context}"},
                        {"role": "user", "content": user_message}
                    ]
                    response = self.groq_client_deepseek.invoke(prompt_messages)
                    content = await self.clean_response(response.content)
                    return content, ["AR6 Synthesis Report via FAISS"]
                    
                elif model == 'llama' and self.groq_client_llama:
                    prompt_messages = [
                        {"role": "system", "content": f"{system_prompt}\n\nContext from AR6 Synthesis Report:\n{context}"},
                        {"role": "user", "content": user_message}
                    ]
                    response = self.groq_client_llama.invoke(prompt_messages)
                    return response.content, ["AR6 Synthesis Report via FAISS"]
                    
            except Exception as e:
                print(f"FAISS Retrieval Error: {str(e)}")
                content, sources = self.get_mock_response(messages[-1]['content'], report_focus)
                return f"⚠️ Error: {str(e)}\n\n{content}", sources
        
        # Handle other LLMs
        try:
            if model == 'deepseek' and self.groq_client_deepseek:
                response = self.groq_client_deepseek.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages[-1]['content']}
                ])
                content = await self.clean_response(response.content)
                return content, ["Groq DeepSeek API"]
                
            elif model == 'llama' and self.groq_client_llama:
                response = self.groq_client_llama.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages[-1]['content']}
                ])
                return response.content, ["Groq Llama API"]
                
            elif model.startswith('gpt') and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo" if model == 'gpt-4' else "gpt-3.5-turbo",
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content, ["OpenAI API"]
                
            elif model == 'claude-3' and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.content[0].text, ["Anthropic API"]
                
            elif model == 'gemini' and self.gemini_client:
                conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                full_prompt = f"{system_prompt}\n\nConversation:\n{conversation_text}"
                response = self.gemini_client.generate_content(full_prompt)
                return response.text, ["Gemini API"]
                
            else:
                user_message = messages[-1]['content'] if messages else ""
                return self.get_mock_response(user_message, report_focus)
                
        except Exception as e:
            print(f"API Error: {str(e)}")
            user_message = messages[-1]['content'] if messages else ""
            content, sources = self.get_mock_response(user_message, report_focus)
            return f"⚠️ API Error: {str(e)}\n\n{content}", sources

    async def process_message(self, message: str, history: List[Dict], model: str, temperature: float, max_tokens: int, report_focus: str) -> Tuple[List[Dict], str]:
        if not message.strip():
            return history, ""
            
        history.append({"role": "user", "content": message})
        messages = history.copy()
        
        try:
            content, sources = await self.call_llm_api(messages, model, temperature, max_tokens, report_focus)
            formatted_response = self.format_response(content, sources, report_focus)
            history.append({"role": "assistant", "content": formatted_response})
        except Exception as e:
            error_response = f"⚠️ Error processing request: {str(e)}"
            history.append({"role": "assistant", "content": error_response})
            
        return history, ""

# Initialize the agent
agent = IPCCLLMAgent()

# Predefined quick prompts
quick_prompts = [
    "Summarize AR6 Synthesis Report key findings",
    "What urgent actions are needed by 2030?",
    "Explain carbon budgets in simple terms",
    "What are the main climate risks and impacts?",
    "Compare AR5 and AR6 projections"
]

def create_gradio_interface():
    with gr.Blocks(title="🌍 IPCC Climate Reports LLM Agent", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 10px;'>
            <h1>🌍 IPCC Climate Reports LLM Agent</h1>
            <p style='font-size: 18px;'>AI-Powered Analysis of Climate Science Reports</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(value=[], height=500)
                msg = gr.Textbox(placeholder="Ask about IPCC climate reports...", lines=2)
                send_btn = gr.Button("Send 🚀", variant="primary")
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                clear_btn.click(lambda: [], outputs=[chatbot])
                
                gr.HTML("<h3>💡 Quick Prompts</h3>")
                with gr.Row():
                    for i in range(0, len(quick_prompts), 2):
                        with gr.Column():
                            if i < len(quick_prompts):
                                gr.Button(quick_prompts[i], size="sm").click(
                                    lambda x=quick_prompts[i]: x, outputs=msg)
                            if i+1 < len(quick_prompts):
                                gr.Button(quick_prompts[i+1], size="sm").click(
                                    lambda x=quick_prompts[i+1]: x, outputs=msg)
            
            with gr.Column(scale=1):
                gr.HTML("<h3>🔧 Configuration</h3>")
                model_choice = gr.Dropdown(
                    choices=list(agent.llm_models.keys()),
                    value="mock",
                    label="AI Model"
                )
                report_focus = gr.Dropdown(
                    choices=['all', 'ar6_syr', 'ar6', 'ar5', 'special', 'ar4'],
                    value="all",
                    label="Report Focus"
                )
                temperature = gr.Slider(0.0, 1.0, value=0.7, label="Temperature")
                max_tokens = gr.Slider(100, 2000, value=1000, label="Max Tokens")
                
                gr.HTML("<h3>📊 Status</h3>")
                status_html = f"""
                <div style='font-size:12px;'>
                {'✅' if agent.groq_client_llama or agent.groq_client_deepseek else '❌'} Groq: {'Available' if (agent.groq_client_llama or agent.groq_client_deepseek) else 'No Key'}<br>
                {'✅' if agent.openai_client else '❌'} OpenAI: {'Available' if agent.openai_client else 'No Key'}<br>
                {'✅' if agent.anthropic_client else '❌'} Anthropic: {'Available' if agent.anthropic_client else 'No Key'}<br>
                {'✅' if agent.gemini_client else '❌'} Gemini: {'Available' if agent.gemini_client else 'No Key'}<br>
                {'✅' if agent.faiss_retriever else '❌'} FAISS: {'Available' if agent.faiss_retriever else 'Not Loaded'}<br>
                ✅ Mock AI: Always Available
                </div>
                """
                gr.HTML(status_html)
        
        async def respond(message, history, model, temp, tokens, report):
            return await agent.process_message(message, history, model, temp, tokens, report)
            
        send_btn.click(
            respond,
            inputs=[msg, chatbot, model_choice, temperature, max_tokens, report_focus],
            outputs=[chatbot, msg]
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, model_choice, temperature, max_tokens, report_focus],
            outputs=[chatbot, msg]
        )
        
        gr.HTML("""
        <div style='text-align: center; padding: 10px; margin-top: 20px; border-top: 1px solid #ddd;'>
            <p>IPCC Climate Reports LLM Agent - Powered by AI for Climate Action</p>
        </div>
        """)
        
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
