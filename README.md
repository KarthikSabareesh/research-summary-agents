# Research Summary Agents - Full Stack Application

AI-powered research assistant with **Hybrid Search (BM25 + Semantic)** and **Citation Management**.

Built with:
- **Backend**: Flask REST API
- **Frontend**: React
- **AI Agents**: LangGraph + Google Gemini
- **Search**: Tavily API (web search)
- **Vector Database**: Pinecone (semantic search)
- **Keyword Search**: BM25 Retriever
- **Safety**: Custom guardrails implementation

---

## Features

### 🔍 Research Agent
- **Hybrid Search**: Combines BM25 (keyword) + semantic search for optimal retrieval
- **Web Search**: Real-time information via Tavily API
- **Dynamic Knowledge Base**: Self-building from search results

### 📚 Citation Management
- **Simple Format**: Clean citation formatting
- **Automatic Deduplication**: Same URL = same citation number
- **Professional Output**: Publication-ready citations

### 🎯 Summary Agent
- **Executive Summaries**: Structured output with key findings
- **Citation Integration**: Properly cited sources
- **Confidence Assessment**: High, medium, or low confidence levels

### 🛡️ Safety & Guardrails
- PII Detection (emails, phones, SSNs, credit cards, API keys)
- Jailbreak Detection (prompt injection, instruction override)
- Content Safety (violence, hate speech, illegal activities)
- Multi-layer validation (input, research output, final summary)
  
---

# Execution Run Video:
https://drive.google.com/file/d/1RWOr5Z9GGzmDB79_L0kkgc58rITqqjoI/view?usp=sharing

---

## Setup Instructions

### Prerequisites

1. **Python 3.10+**
2. **Node.js 16+** and npm
3. **API Keys**:
   - `PINECONE_API_KEY` - [Get from Pinecone](https://www.pinecone.io/)
   - `GOOGLE_API_KEY` - [Get from Google AI Studio](https://makersuite.google.com/app/apikey)
   - `TAVILY_API_KEY` - [Get from Tavily](https://tavily.com/)

### Step 1: Clone & Setup Environment

```bash
cd research-summary-agents

# Create .env file with your API keys
cat > .env << EOF
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=research-agents
GOOGLE_API_KEY=your-google-api-key
TAVILY_API_KEY=your-tavily-api-key
EOF
```

### Step 2: Install Python Dependencies

```bash
# Install main project dependencies
pip install langchain langchain-community langchain-google-genai
pip install langgraph langchain-pinecone pinecone-client
pip install langchain-huggingface sentence-transformers
pip install tavily-python beautifulsoup4 pydantic

# Install backend-specific dependencies
pip install flask flask-cors gunicorn python-dotenv
```

### Step 3: Setup Pinecone Index

The index will be created automatically when you first run the application. The system uses:
- **Model**: `intfloat/e5-large-v2` (1024 dimensions)
- **Metric**: Cosine similarity

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application (locally)

**Terminal 1 - Backend:**
```bash
# Make sure you're in the project root with .env file
export $(cat .env | xargs)  # Load environment variables

cd backend
python app.py
```

Backend will start on: `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Frontend will open automatically at: `http://localhost:3000`

## Usage

### 1. Start Both Servers
Follow the "Running the Application" section above.

### 2. Access the Web Interface
Open `http://localhost:3000` in your browser

### 3. Submit a Query
- Enter your research question in the text area
- Click "Research" button

### 4. View Results
The interface has 3 tabs:
- **📄 Summary**: Key findings and executive summary
- **📚 Citations**: Formatted citations with credibility scores
- **🔍 Full Report**: Complete raw output

### Example Queries
- "What are the latest developments in quantum computing?"
- "Who is the president of India?"
- "What are the benefits and risks of AI in healthcare?"
- "Explain the current state of electric vehicle adoption"

---
## Future direction
- **Include MCP** connectors for ArXiv, Google Drive, Notion for more selective research
- **Scale up with cloud** providers like AWS or Azure
- **Integrate Kubernetes**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Flask + Flask-CORS |
| Frontend | React 18 |
| AI Orchestration | LangGraph |
| LLM | Google Gemini 2.5 Flash |
| Vector DB | Pinecone (Serverless) |
| Embeddings | HuggingFace `e5-large-v2` |
| Web Search | Tavily API |
| Keyword Search | BM25Retriever |
| Safety | Custom Guardrails |
| Citations | Custom Manager |

---

**Built with ❤️**
