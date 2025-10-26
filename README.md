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

## Project Structure

```
research-summary-agents/
├── research-summary-agents.py    # Core agent implementation
├── backend/
│   ├── app.py                    # Flask REST API
│   └── requirements.txt          # Backend dependencies
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js               # Main React component
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   └── components/
│   │       ├── QueryForm.js     # Search input form
│   │       ├── ResultsDisplay.js # Results with tabs
│   │       ├── SystemStatus.js   # Status indicator
│   │       ├── CitationsList.js  # Citations display
│   │       └── *.css            # Component styles
│   └── package.json
└── README.md
```

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

## Running the Application

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
Open `http://localhost:3000` in your browser.

### 3. Submit a Query
- Enter your research question in the text area
- Select citation format (APA, MLA, Chicago, or Simple)
- Click "Research" button

### 4. View Results
The interface has 4 tabs:
- **📄 Summary**: Key findings and executive summary
- **🎯 Credibility Analysis**: Source quality charts and statistics
- **📚 Citations**: Formatted citations with credibility scores
- **🔍 Full Report**: Complete raw output

### Example Queries
- "What are the latest developments in quantum computing?"
- "Who is the president of India?"
- "What are the benefits and risks of AI in healthcare?"
- "Explain the current state of electric vehicle adoption"

---

## Troubleshooting

### Backend Issues

**"System not initialized" error:**
- Check if all API keys are set in `.env`
- Verify Pinecone index exists or can be created
- Check console for initialization errors

**Slow queries:**
- Normal for first query (model loading)
- Subsequent queries use cached models
- Complex queries may take 30-60 seconds

### Frontend Issues

**"No response from server":**
- Ensure backend is running on port 5000
- Check CORS is enabled in Flask
- Verify firewall allows localhost:5000

**Build errors:**
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again
- Ensure Node.js version is 16+

### API Key Issues

**Pinecone errors:**
- Verify API key is valid
- Check index name matches `.env`
- Ensure free tier hasn't exceeded limits

**Tavily errors:**
- System falls back to knowledge-base only mode
- Citations won't be generated
- Check API key and quota

---

## Performance Tips

1. **First Run**: Takes longer due to model downloads
2. **Caching**: Models are cached after first use
3. **Concurrent Queries**: Use different `thread_id` values
4. **Knowledge Base**: Grows over time, improving future query speed

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

**Built with ❤️ using LangGraph, Gemini, Pinecone, and Tavily**
