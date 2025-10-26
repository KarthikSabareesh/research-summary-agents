import os
import re
from io import StringIO
from typing import Annotated, Literal, TypedDict
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import re

GUARDRAILS_ENABLED = True

class CustomGuard:
    """Custom safety guard with PII detection, jailbreak detection, and content filtering."""

    @staticmethod
    def detect_pii(text: str) -> tuple[bool, list[str]]:
        """Detect PII in text using regex patterns."""
        pii_found = []

        # Email detection
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            pii_found.append("EMAIL")

        # Phone number
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            pii_found.append("PHONE")

        # SSN pattern
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            pii_found.append("SSN")

        # Credit card
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
            pii_found.append("CREDIT_CARD")

        # API keys or secrets
        if re.search(r'\b(sk-[a-zA-Z0-9]{32,}|pk-[a-zA-Z0-9]{32,})\b', text):
            pii_found.append("API_KEY")

        return len(pii_found) > 0, pii_found

    @staticmethod
    def detect_jailbreak(text: str) -> tuple[bool, str]:
        """Detect jailbreak attempts."""
        jailbreak_patterns = [
            r"do anything now",
            r"from now on.{0,50}act as",
            r"you are going to act as",
            r"pretend (you are|to be)",
            r"roleplay as",
            r"simulate being",

            r"ignore (all )?previous (instructions|rules|prompts)",
            r"disregard (all )?(previous|above|earlier)",
            r"forget (everything|all|your) (you know|instructions|rules)",
            r"new instructions:",
            r"system prompt:",
            r"override (instructions|rules|system)",

            r"developer mode",
            r"admin mode",
            r"god mode",
            r"jailbreak mode",

            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"\[SYSTEM\]",
            r"\[INST\]",
        ]

        text_lower = text.lower()
        for pattern in jailbreak_patterns:
            if re.search(pattern, text_lower):
                return True, f"Jailbreak pattern detected"

        return False, ""

    @staticmethod
    def detect_unsafe_content(text: str) -> tuple[bool, str]:
        """Detect unsafe content including violence, hate speech, and harmful content."""

        text_lower = text.lower()

        violence_keywords = [
            "genocide", "ethnic cleansing", "mass killing", "exterminate",
            "bomb", "weapon", "gun", "explosive", "grenade",
            "kill", "murder", "assassinate", "torture", "massacre",
            "slaughter", "execute", "lynch", "eradicate",
        ]

        for keyword in violence_keywords:
            if keyword in text_lower:
                return True, f"Violent content detected: '{keyword}'"

        hate_speech_patterns = [
            # Dehumanizing language
            r"\b(scum|vermin|animals|insects|rats|parasites|trash|garbage|filth)\b",
            r"(don't|do not|doesn't) deserve to (live|breathe|exist)",
            r"should (die|be killed|be eliminated|be exterminated|not exist)",

            # Group-targeted violence
            r"(kill|eliminate|exterminate|eradicate) all (the )?\w+ (people|race|group)",
            r"genocide (all |the )?\w+ (people|race)",
            r"how to (kill|eliminate|harm|attack) (all |the )?\w+ (people|race|group)",

            # Hate group terminology
            r"\b(nazi|nazism|fascis[mt]|kkk|aryan nation)\w*\b",
            r"\b(white supremac|racial purity|master race)\w*\b",

            # Explicit hate/discrimination
            r"\b(hate|despise) (all |the )?\w+ (people|race|religion|group)",
            r"\w+ (people|race|group) (are|is) (inferior|subhuman|animals|scum)",
        ]

        for pattern in hate_speech_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return True, f"Hate speech pattern detected: '{match.group()}'"

        illegal_keywords = [
            "illegal drugs", "drug trafficking", "fraud", "scam",
            "money laundering", "stolen", "hack into", "credit card theft",
        ]

        for keyword in illegal_keywords:
            if keyword in text_lower:
                return True, f"Illegal activity content detected: '{keyword}'"

        self_harm_keywords = [
            "suicide", "self-harm", "kill yourself", "end your life",
        ]

        for keyword in self_harm_keywords:
            if keyword in text_lower:
                return True, f"Self-harm content detected: '{keyword}'"

        return False, ""

    def validate(self, text: str):
        """Validate text for PII, jailbreaks, and unsafe content."""
        class ValidationResult:
            def __init__(self, passed: bool, error: str = None):
                self.validation_passed = passed
                self.error = error
                self.validated_output = text

        # Check for PII
        has_pii, pii_types = self.detect_pii(text)
        if has_pii:
            return ValidationResult(False, f"PII detected: {', '.join(pii_types)}")

        # Check for jailbreak attempts
        is_jailbreak, jb_msg = self.detect_jailbreak(text)
        if is_jailbreak:
            return ValidationResult(False, jb_msg)

        # Check for unsafe content
        is_unsafe, unsafe_msg = self.detect_unsafe_content(text)
        if is_unsafe:
            return ValidationResult(False, unsafe_msg)

        return ValidationResult(True)

guard = CustomGuard()

# -----------------------------
# Citation Manager
# -----------------------------
class CitationManager:
    """
    Manages citations with multiple formatting styles and deduplication.
    """

    def __init__(self):
        self.citations = []
        self.url_index = {}  # Track URLs to avoid duplicates

    def add_citation(self, url: str, title: str = "", author: str = "",
                     date: str = "", credibility_score: int = 0,
                     accessed_date: str = None) -> int:

        if url in self.url_index:
            return self.url_index[url]

        citation_number = len(self.citations) + 1

        citation = {
            "number": citation_number,
            "url": url,
            "title": title or self._extract_title_from_url(url),
            "author": author or "Unknown",
            "date": date or "n.d.",
            "accessed_date": accessed_date or datetime.now().strftime("%Y-%m-%d"),
            "credibility_score": credibility_score,
            "domain": urlparse(url).netloc if url.startswith('http') else url
        }

        self.citations.append(citation)
        self.url_index[url] = citation_number

        return citation_number

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/').split('/')[-1]
            # Remove extension and convert hyphens/underscores to spaces
            title = path.replace('-', ' ').replace('_', ' ').replace('.html', '').replace('.htm', '')
            return title.title() if title else parsed.netloc
        except:
            return url

    def format_citation(self, citation_number: int, style: str = "simple") -> str:
        if citation_number < 1 or citation_number > len(self.citations):
            return f"[Invalid citation: {citation_number}]"

        cit = self.citations[citation_number - 1]  
        # Simple format
        return f"[{citation_number}] {cit['title']} - {cit['domain']}\n    {cit['url']}"

    def get_all_citations(self, style: str = "simple") -> list[str]:
        return [self.format_citation(i+1, style) for i in range(len(self.citations))]

    def clear(self):
        """Clear all citations."""
        self.citations = []
        self.url_index = {}

citation_manager = CitationManager()

# -----------------------------
# Knowledge Base: Dynamically populated from search results
# -----------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Text splitter for processing search results before storing
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)

# -----------------------------
# Pinecone vector store (serverless, v6 SDK)
# -----------------------------
from pinecone import Pinecone, ServerlessSpec                  
from langchain_pinecone import PineconeVectorStore               
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever  
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    encode_kwargs={"normalize_embeddings": True}
)
EMBED_DIM = len(embeddings.embed_query("dimension probe"))
index_name = os.environ["PINECONE_INDEX_NAME"]


existing = {idx.name for idx in pc.list_indexes().indexes}  # -> names of existing indexes
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Build vector store
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# Create semantic retriever (MMR)
semantic_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

# Initialize BM25 retriever (keyword-based search)
# Start with empty document list - will be populated dynamically
bm25_documents = []
bm25_retriever = BM25Retriever.from_documents(
    bm25_documents if bm25_documents else [Document(page_content="initialization", metadata={"type": "init"})],
    k=3
)

# Create hybrid retriever combining BM25 (keyword) + semantic search
# Weights can be adjusted based on your use case
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5],  # Equal weight to keyword and semantic search
)


# -----------------------------
# Retriever tool (Hybrid Search: BM25 + Semantic)
# -----------------------------
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_search_data",
    "Hybrid search combining keyword (BM25) and semantic search to find relevant information from the knowledge base.",
    document_separator="\n\n---\n\n",
    response_format="content",
)

# -----------------------------
# Tavily Search Tool (Web Search for Research Agent)
# -----------------------------
try:
    from langchain_community.tools.tavily_search import TavilySearchResults

    # TAVILY_API_KEY should be set in your shell
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    TAVILY_ENABLED = True
    print("Tavily search initialized successfully")
except ImportError:
    print("Warning: Tavily not installed. Install with: pip install langchain-community tavily-python")
    TAVILY_ENABLED = False
    tavily_tool = None
except Exception as e:
    print(f"Warning: Tavily initialization failed: {e}")
    print("Make sure TAVILY_API_KEY is set in your environment")
    TAVILY_ENABLED = False
    tavily_tool = None


# -----------------------------
# LLMs
# -----------------------------
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Use Gemini for both agents
research_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
summary_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


# -----------------------------
# AGENT 1: Research Agent
# -----------------------------
class GuardrailsViolationError(Exception):
    """Raised when guardrails validation fails."""
    pass

def apply_guardrails(text: str, context: str = "input") -> str:
    """
    Apply guardrails to validate input/output safety.

    Raises:
        GuardrailsViolationError: If validation fails (PII, jailbreak, unsafe content detected)
    """
    if not GUARDRAILS_ENABLED or guard is None:
        return text

    try:
        result = guard.validate(text)
        if result.validation_passed:
            return text
        else:
            error_msg = f"Guardrails {context} validation failed: {result.error}"
            print(f"\n⛔ {error_msg}")
            raise GuardrailsViolationError(error_msg)
    except GuardrailsViolationError:
        raise
    except Exception as e:
        error_msg = f"Guardrails {context} error: {e}"
        print(f"\n⛔ {error_msg}")
        raise GuardrailsViolationError(error_msg)

def research_agent(state: MessagesState):
    """
    Research Agent: Intelligent multi-source information gathering.
    """
    messages = state["messages"]
    user_query = messages[0].content if messages else ""

    # Reset citation manager for new query
    global citation_manager
    print(f"[Research Agent] Clearing citation manager (had {len(citation_manager.citations)} citations)")
    citation_manager.clear()
    print(f"[Research Agent] Citation manager cleared (now has {len(citation_manager.citations)} citations)")

    try:
        # Apply input guardrails (will raise exception if validation fails)
        safe_query = apply_guardrails(user_query, "input")
    except GuardrailsViolationError as e:
        # Guardrails blocked the input - return error message and STOP
        error_message = f"""
⛔ REQUEST BLOCKED BY SAFETY CONTROLS ⛔

Your query was blocked for safety reasons:
{str(e)}

This system has guardrails to protect against:
- Personal Identifiable Information (PII): emails, phone numbers, SSNs, credit cards
- Jailbreak attempts: prompt injection, instruction override
- Unsafe content: violence, hate speech, illegal activities

Please rephrase your query without sensitive information.
"""
        return {
            "messages": [
                AIMessage(
                    content=error_message,
                    name="research_agent",
                    additional_kwargs={"agent": "research", "blocked": True, "reason": str(e)}
                )
            ]
        }

    print(f"\n[Research Agent] Processing query: {safe_query[:100]}...")

    # STEP 1: Check knowledge base first
    print("[Research Agent] Step 1: Checking knowledge base...")
    kb_results = ""
    try:
        kb_results = retriever_tool.invoke({"query": safe_query})
        if kb_results and kb_results.strip():
            print(f"[Research Agent] ✓ Found {len(kb_results)} chars in knowledge base")
        else:
            print("[Research Agent] ○ Knowledge base is empty or no relevant results")
    except Exception as e:
        print(f"[Research Agent] ○ Knowledge base query failed: {e}")
        kb_results = ""

    # STEP 2: Evaluate if knowledge base results are sufficient
    need_web_search = True
    if kb_results and kb_results.strip():
        print("[Research Agent] Step 2: Evaluating knowledge base sufficiency...")

        # STEP 2A: Quick keyword relevance check (fast filter)
        query_terms = set(re.findall(r'\b\w{4,}\b', safe_query.lower()))
        kb_lower = kb_results.lower()
        matching_terms = [term for term in query_terms if term in kb_lower]
        match_ratio = len(matching_terms) / max(len(query_terms), 1)

        print(f"[Research Agent]   Keyword check: {len(matching_terms)}/{len(query_terms)} terms matched ({match_ratio:.1%})")

        if match_ratio < 0.75:
            print("[Research Agent] ○ Low keyword overlap - KB not relevant, will perform web search")
            need_web_search = True
        else:
            # STEP 2B: Borderline case - use strict LLM evaluation
            print("[Research Agent]   Sufficient keyword overlap - checking with LLM...")

            eval_prompt = f"""Strict relevance evaluation.

USER QUERY: "{safe_query}"

KNOWLEDGE BASE CONTENT:
{kb_results[:1200]}

STRICT EVALUATION RULES:
❌ If query asks about Person/Entity A but KB only has Person/Entity B → INSUFFICIENT
❌ If query asks about "Company X" but KB discusses "Company Y" → INSUFFICIENT
❌ If query asks about "2024" but KB only has "2023" data → INSUFFICIENT
✅ Only answer SUFFICIENT if KB directly discusses the EXACT subject/entity in the query

Question: Does the knowledge base contain DIRECT information about the SPECIFIC subject asked in the query?

Answer with ONE WORD only (SUFFICIENT or INSUFFICIENT):"""

            eval_response = research_model.invoke([HumanMessage(content=eval_prompt)])
            eval_decision = eval_response.content.strip().upper()

            if "SUFFICIENT" in eval_decision:
                need_web_search = False
                print("[Research Agent] ✓ Knowledge base is sufficient - skipping web search")
            else:
                print("[Research Agent] ○ Knowledge base insufficient - will perform web search")
    else:
        print("[Research Agent] Step 2: Knowledge base empty - will perform web search")

    # STEP 3: Perform web search if needed (with citations)
    web_results = ""
    documents_to_store = []
    credibility_scores = []

    if need_web_search and TAVILY_ENABLED and tavily_tool:
        print("[Research Agent] Step 3: Performing web search via Tavily...")
        try:
            tavily_results = tavily_tool.invoke({"query": safe_query})
            web_results_list = []
            print(f"[Research Agent] ✓ Retrieved web search results")

            # Process and score each search result
            try:
                if isinstance(tavily_results, list):
                    print("[Research Agent] Step 3a: Scoring source credibility...")
                    for idx, item in enumerate(tavily_results):
                        if isinstance(item, dict):
                            content = item.get('content', '') or item.get('snippet', '')
                            url = item.get('url', 'unknown')
                            title = item.get('title', '')

                            if content and url != 'unknown':
                                # Score source credibility
                                credibility = credibility_scorer.score_source(
                                    url=url,
                                    content=content,
                                    date_str=None  # Tavily doesn't always provide dates
                                )
                                credibility_scores.append(credibility)

                                # Add citation
                                citation_num = citation_manager.add_citation(
                                    url=url,
                                    title=title,
                                    credibility_score=credibility['score']
                                )
                                print(f"[Research Agent] Added citation #{citation_num}: {title[:50]}... (Total: {len(citation_manager.citations)})")

                                # Create document with enhanced metadata
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": url,
                                        "query": safe_query,
                                        "type": "tavily_search",
                                        "title": title,
                                        "citation_number": citation_num,
                                        "credibility_score": credibility['score'],
                                        "credibility_tier": credibility['tier'],
                                        "credibility_confidence": credibility['confidence_level'],
                                        "domain": credibility['domain']
                                    }
                                )
                                documents_to_store.append(doc)

                                # Add to web results with citation
                                web_results_list.append(
                                    f"[Source {citation_num}] [{credibility['tier']} - Score: {credibility['score']}/100]\n"
                                    f"URL: {url}\n"
                                    f"Content: {content}\n"
                                )

                                print(f"[Research Agent]   Source {citation_num}: {credibility['domain']} - "
                                      f"Credibility: {credibility['score']}/100 ({credibility['tier']})")

                    web_results = "\n---\n".join(web_results_list)
                    print(f"[Research Agent] ✓ Scored {len(credibility_scores)} sources")

                elif isinstance(tavily_results, str) and tavily_results.strip():
                    # Fallback for string results
                    doc = Document(
                        page_content=tavily_results,
                        metadata={"source": "tavily", "query": safe_query, "type": "tavily_search"}
                    )
                    documents_to_store.append(doc)
                    web_results = tavily_results

            except Exception as e:
                print(f"[Research Agent] Warning: Could not parse/score Tavily results: {e}")
                web_results = str(tavily_results)  # Fallback to raw results

        except Exception as e:
            print(f"[Research Agent] ○ Web search failed: {e}")
            web_results = ""
    elif need_web_search and not TAVILY_ENABLED:
        print("[Research Agent] ⚠ Web search needed but Tavily is not enabled")

    # STEP 4: Store new documents in both Pinecone (semantic) and BM25 (keyword)
    if documents_to_store:
        try:
            doc_chunks = text_splitter.split_documents(documents_to_store)

            # Add to Pinecone vector store (semantic search)
            vectorstore.add_documents(doc_chunks)

            # Add to BM25 retriever (keyword search)
            # BM25 needs to be rebuilt with all documents
            global bm25_retriever
            existing_docs = getattr(bm25_retriever, 'docs', [])
            # Filter out initialization doc if present
            existing_docs = [doc for doc in existing_docs if doc.metadata.get("type") != "init"]
            all_docs = existing_docs + doc_chunks
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=3)

            # Update the ensemble retriever with the new BM25 retriever
            global retriever
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5],
            )

            print(f"[Research Agent] ✓ Stored {len(doc_chunks)} chunks in hybrid knowledge base (BM25 + semantic)")
        except Exception as e:
            print(f"[Research Agent] Warning: Could not store documents: {e}")

    # STEP 5: Compile comprehensive research findings
    print("[Research Agent] Step 4: Compiling research findings...")

    research_prompt = f"""You are a Research Agent compiling findings from multiple sources.

User Query: {safe_query}

Knowledge Base Results:
{kb_results if kb_results else "No knowledge base results"}

{"="*50}

Web Search Results:
{web_results if web_results else "No web search performed"}

{"="*50}

Task: Synthesize the above information into a comprehensive research report.
Include:
- All relevant findings from both sources
- Clear citations (mention if from "knowledge base" or "web search")
- Note which information sources were used

Your findings will be passed to a Summary Agent for final formatting."""

    response = research_model.invoke([HumanMessage(content=research_prompt)])
    research_findings = response.content

    # Apply output guardrails (will raise exception if LLM output violates safety rules)
    try:
        safe_findings = apply_guardrails(research_findings, "output")
    except GuardrailsViolationError as e:
        # Research output violated guardrails - return sanitized error
        error_message = f"""
⛔ RESEARCH OUTPUT BLOCKED BY SAFETY CONTROLS ⛔

The research findings contained unsafe content and were blocked:
{str(e)}

The query has been rejected. Please try a different query.
"""
        return {
            "messages": [
                AIMessage(
                    content=error_message,
                    name="research_agent",
                    additional_kwargs={"agent": "research", "blocked": True, "reason": str(e)}
                )
            ]
        }

    print("[Research Agent] ✓ Research complete\n")

    # Get credibility summary
    cred_summary = citation_manager.get_credibility_summary()

    # Store research findings in state with metadata
    print(f"[Research Agent] Completing with {len(citation_manager.citations)} citations in manager")
    print(f"[Research Agent] Citation manager ID: {id(citation_manager)}")

    return {
        "messages": [
            AIMessage(
                content=safe_findings,
                name="research_agent",
                additional_kwargs={
                    "agent": "research",
                    "query": safe_query,
                    "used_kb": bool(kb_results),
                    "used_web_search": bool(web_results),
                    "kb_sufficient": not need_web_search,
                    "credibility_summary": cred_summary,
                    "total_sources": len(credibility_scores),
                    "citations_count": len(citation_manager.citations)
                }
            )
        ]
    }


# -----------------------------
# AGENT 2: Summary Agent
# -----------------------------
from pydantic import BaseModel, Field
from typing import Literal

class SourceCredibilityInfo(BaseModel):
    """Source credibility information."""
    average_score: float = Field(description="Average credibility score of all sources")
    high_quality_sources: int = Field(description="Number of high-quality sources (>70)")
    medium_quality_sources: int = Field(description="Number of medium-quality sources (40-70)")
    low_quality_sources: int = Field(description="Number of low-quality sources (<40)")
    total_sources: int = Field(description="Total number of sources evaluated")

class ExecutiveSummary(BaseModel):
    """Structured executive summary output with credibility tracking."""
    title: str = Field(description="Brief title for the research")
    key_findings: list[str] = Field(description="3-5 key findings from the research")
    summary: str = Field(description="Concise executive summary (2-3 paragraphs)")
    sources: list[str] = Field(description="List of sources cited with credibility scores")
    confidence_level: str = Field(description="Confidence level: high, medium, or low")
    source_credibility: Optional[SourceCredibilityInfo] = Field(
        default=None,
        description="Credibility analysis of sources used"
    )

def summary_agent(state: MessagesState):
    """
    Summary Agent: Converts research findings into executive summary with credibility analysis.
    Demonstrates: Structured output, multi-agent collaboration, synthesis, source credibility
    """
    messages = state["messages"]

    # Get the original query and research findings
    original_query = messages[0].content if messages else ""
    research_findings = messages[-1].content if len(messages) > 1 else ""

    # Get credibility information from research agent
    credibility_summary = None
    citations_count = 0
    if len(messages) > 1 and hasattr(messages[-1], 'additional_kwargs'):
        additional_kwargs = messages[-1].additional_kwargs
        credibility_summary = additional_kwargs.get('credibility_summary', None)
        citations_count = additional_kwargs.get('citations_count', 0)

        # Check if research agent already blocked the request
        if additional_kwargs.get('blocked', False):
            # Research was already blocked - just pass through the block message
            return {
                "messages": [
                    AIMessage(
                        content=research_findings,  # This contains the block message
                        name="summary_agent",
                        additional_kwargs={"agent": "summary", "blocked": True, "passthrough": True}
                    )
                ]
            }

    # Summary agent system prompt
    system_prompt = """You are a Summary Agent specialized in creating executive summaries with source credibility analysis.

Your task:
1. Analyze the research findings provided by the Research Agent
2. Extract key insights and important information
3. Create a concise, well-structured executive summary
4. Organize information in a clear, actionable format
5. Cite sources with their credibility scores (use [Source N] notation)
6. Prioritize information from higher-credibility sources

Format your summary for business/executive audience with:
- Clear title
- Key findings (bullet points with source citations)
- Executive summary (2-3 paragraphs)
- Source citations with credibility indicators
- Confidence assessment based on source quality"""

    summary_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Original Query: {original_query}

Research Findings:
{research_findings}

Please create an executive summary of these research findings.""")
    ]

    # Use structured output for consistent formatting
    model_with_structure = summary_model.with_structured_output(ExecutiveSummary)
    summary_result = model_with_structure.invoke(summary_messages)

    # Format the executive summary
    formatted_summary = f"""
{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Title: {summary_result.title}

KEY FINDINGS:
"""
    for i, finding in enumerate(summary_result.key_findings, 1):
        formatted_summary += f"{i}. {finding}\n"

    formatted_summary += f"""
SUMMARY:
{summary_result.summary}
"""

    # Add formatted citations
    formatted_summary += f"""
CITATIONS:
{'─'*80}
"""
    if citations_count > 0:
        citations = citation_manager.get_all_citations(style="simple")
        for citation in citations:
            formatted_summary += f"{citation}\n\n"
    else:
        formatted_summary += "No citations available (knowledge base used)\n"

    formatted_summary += f"""
Confidence Level: {summary_result.confidence_level.upper()}
{'='*80}
"""

    # Apply final output guardrails
    try:
        safe_summary = apply_guardrails(formatted_summary, "final_output")
    except GuardrailsViolationError as e:
        # Summary output violated guardrails - return error
        error_message = f"""
⛔ SUMMARY BLOCKED BY SAFETY CONTROLS ⛔

The executive summary contained unsafe content and was blocked:
{str(e)}

The query has been rejected. Please try a different query.
"""
        return {
            "messages": [
                AIMessage(
                    content=error_message,
                    name="summary_agent",
                    additional_kwargs={"agent": "summary", "blocked": True, "reason": str(e)}
                )
            ]
        }

    return {
        "messages": [
            AIMessage(
                content=safe_summary,
                name="summary_agent",
                additional_kwargs={
                    "agent": "summary",
                    "structured_data": summary_result.dict()
                }
            )
        ]
    }


# -----------------------------
# LangGraph Multi-Agent Workflow
# -----------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Create the workflow graph
workflow = StateGraph(MessagesState)

# Add agent nodes
workflow.add_node("research_agent", research_agent)
workflow.add_node("summary_agent", summary_agent)

# Define the workflow: User Query -> Research Agent -> Summary Agent -> End
workflow.add_edge(START, "research_agent")
workflow.add_edge("research_agent", "summary_agent")
workflow.add_edge("summary_agent", END)

# Compile the graph with memory checkpointing for conversation history
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

print("\nMulti-Agent Research Assistant initialized successfully!")
print("- Research Agent: Web search (Tavily) + Hybrid RAG (BM25 + Semantic)")
print("- Summary Agent: Executive summary generation with credibility analysis")
print(f"- Guardrails: {'Enabled' if GUARDRAILS_ENABLED else 'Disabled'}")
print(f"- Tavily Search: {'Enabled' if TAVILY_ENABLED else 'Disabled'}")
print("- Retrieval: Hybrid search with 50% BM25 (keyword) + 50% semantic")
print("- Source Credibility: 100-point scoring system (domain + freshness + quality)")
print("- Citations: APA, MLA, Chicago, Simple formats with automatic deduplication")


# -----------------------------
# Main Execution: Multi-Agent Research Assistant Demo
# -----------------------------
from langchain_core.messages import BaseMessage

def run_research_query(query: str, thread_id: str = "demo-1", verbose: bool = True):
    """
    Run a research query through the multi-agent system.

    Args:
        query: User's research question
        thread_id: Thread ID for conversation memory
        verbose: Print intermediate agent outputs
    """
    print(f"\n{'='*80}")
    print(f"Research Query: {query}")
    print(f"{'='*80}\n")

    config = {"configurable": {"thread_id": thread_id}}

    # Stream the graph execution
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    ):
        for node_name, update in chunk.items():
            msg = update["messages"][-1]

            if verbose:
                print(f"\n[{node_name.upper()}]")
                print("-" * 80)

            # Extract content
            if isinstance(msg, BaseMessage):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = getattr(msg, "content", "")

            if verbose and content:
                # Print abbreviated output for research agent
                if node_name == "research_agent":
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(content)

    print(f"\n{'='*80}")
    print("Research complete!")
    print(f"{'='*80}\n")



# -----------------------------
# Demo Queries
# -----------------------------
if __name__ == "__main__":
    # Example 1: Query about existing knowledge base
    run_research_query(
        """How many times as John Cena won the WWE Championship?""",
        thread_id="demo-1"
    )

"""
================================================================================
SETUP AND USAGE GUIDE
================================================================================

REQUIREMENTS:
-------------
This multi-agent research assistant demonstrates advanced AI agent capabilities:

1. Design & Orchestration:
   - Two specialized agents working in sequence
   - Research Agent: Gathers information from web and knowledge base
   - Summary Agent: Synthesizes findings into executive summary

2. Tools & Integrations:
   - RAG: Pinecone vector database with DYNAMIC population from searches
   - Web Search: Tavily API for real-time web information
   - Tool Calling: Automatic tool selection and execution
   - Memory: Conversation state + growing knowledge base from search results

3. Real-world Systems:
   - APIs: Tavily search, Pinecone vector DB, Google Gemini
   - Databases: Pinecone serverless vector database (dynamically populated)
   - Knowledge Base: Self-building from search results

4. Guardrails & Safety:
   - Guardrails AI for content validation
   - Toxic language detection
   - Topic restriction enforcement
   - Input/output sanitization

5. Advanced Agent Tech:
   - LangGraph for agent orchestration
   - Structured output with Pydantic
   - Multi-agent collaboration patterns
   - State management and memory


INSTALLATION:
-------------
pip install langchain langchain-community langchain-google-genai
pip install langgraph langchain-pinecone pinecone-client
pip install langchain-huggingface sentence-transformers
pip install tavily-python
pip install guardrails-ai
pip install beautifulsoup4


ENVIRONMENT VARIABLES:
----------------------
Required:
  export PINECONE_API_KEY="your-pinecone-api-key"
  export PINECONE_INDEX_NAME="your-index-name"
  export GOOGLE_API_KEY="your-google-gemini-api-key"

Optional (for full functionality):
  export TAVILY_API_KEY="your-tavily-api-key"
  export LANGSMITH_API_KEY="your-langsmith-api-key"


USAGE:
------
1. Basic usage:
   from ragagent import run_research_query
   run_research_query("Your research question here")

2. With custom thread ID for conversation memory:
   run_research_query("Your question", thread_id="custom-session-1")

3. Silent mode (only show final summary):
   run_research_query("Your question", verbose=False)

4. Access citation manager for custom formatting:
   from ragagent import citation_manager
   # After a query is run:
   apa_citations = citation_manager.get_all_citations(style="apa")
   mla_citations = citation_manager.get_all_citations(style="mla")
   credibility_stats = citation_manager.get_credibility_summary()


ARCHITECTURE:
-------------
User Query
    |
    v
[Research Agent]
    |
    +-- Input Guardrails Validation
    +-- Hybrid Retrieval (BM25 + Semantic)
    |   |
    |   +-- Check Knowledge Base
    |   +-- Evaluate Sufficiency
    |   +-- Web Search via Tavily (if needed)
    |
    +-- Source Credibility Scoring
    |   |
    |   +-- Domain Reputation (40 pts)
    |   +-- Date Freshness (30 pts)
    |   +-- Content Quality (30 pts)
    |
    +-- Citation Management
    |   |
    |   +-- Add Citations (auto-dedup)
    |   +-- Track Credibility Scores
    |
    +-- Store in Knowledge Base (BM25 + Pinecone)
    +-- Output Guardrails Validation
    |
    v
[Summary Agent]
    |
    +-- Structured Output Generation
    +-- Credibility-Aware Synthesis
    +-- Citation Formatting (APA/MLA/Chicago/Simple)
    +-- Source Quality Analysis
    +-- Executive Summary Formatting
    +-- Final Guardrails Validation
    |
    v
Executive Summary with:
  - Key Findings (cited)
  - Summary (2-3 paragraphs)
  - Source Credibility Analysis
  - Formatted Citations
  - Confidence Level


FEATURES DEMONSTRATED:
----------------------
✓ Multi-agent collaboration (Research + Summary agents)
✓ Tool calling (Tavily search, Pinecone retrieval)
✓ Hybrid RAG: BM25 (keyword) + Semantic search with vector database (Pinecone)
✓ Source Credibility Scoring: 100-point system analyzing domain, freshness, quality
✓ Citation Management: APA, MLA, Chicago, Simple formats with deduplication
✓ Memory and state management (LangGraph checkpointing)
✓ API integrations (Tavily, Pinecone, Gemini)
✓ Database integration (Pinecone serverless)
✓ Guardrails and safety controls (Custom implementation)
✓ Structured output (Pydantic models)
✓ Advanced orchestration (LangGraph)
✓ Real-world web search (Tavily)
✓ Ensemble retrieval with configurable BM25/semantic weights
✓ Credibility-aware summarization (prioritizes high-quality sources)


EXAMPLE OUTPUT:
---------------
When you run a query, you'll see:

[Research Agent] Processing query...
[Research Agent] Step 1: Checking knowledge base...
[Research Agent] Step 2: Evaluating knowledge base sufficiency...
[Research Agent] Step 3: Performing web search via Tavily...
[Research Agent] Step 3a: Scoring source credibility...
[Research Agent]   Source 1: nytimes.com - Credibility: 85/100 (Highly Credible)
[Research Agent]   Source 2: techcrunch.com - Credibility: 65/100 (Credible)
[Research Agent] ✓ Scored 5 sources
[Research Agent] ✓ Stored 12 chunks in hybrid knowledge base

[Summary Agent]
================================================================================
EXECUTIVE SUMMARY
================================================================================

Title: [Generated Title]

KEY FINDINGS:
1. [Finding with source citation]
2. [Finding with source citation]
...

SUMMARY:
[2-3 paragraph executive summary]

SOURCE CREDIBILITY ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Average Source Quality: 72.4/100
  • High-Quality Sources (≥70):    3
  • Medium-Quality Sources (40-69): 2
  • Low-Quality Sources (<40):      0
Total Sources Evaluated: 5

CITATIONS:
────────────────────────────────────────────────────────────────────────────────
[1] Article Title - nytimes.com [Credibility: 85/100]
    https://nytimes.com/article-url

[2] Tech Article - techcrunch.com [Credibility: 65/100]
    https://techcrunch.com/article-url

Confidence Level: HIGH
================================================================================


NOTES:
------
- Source credibility scoring runs automatically on all web search results
- Citations are deduplicated automatically (same URL = same citation number)
- You can change citation format by modifying citation_manager.get_all_citations(style="apa")
- Credibility scores are stored in document metadata for future retrieval
- Summary agent prioritizes information from higher-credibility sources
- Guardrails and Tavily are optional but recommended
- Without Tavily, system falls back to RAG-only mode (no credibility scoring)
- Pinecone and Gemini are required components
- Memory persists across queries within same thread_id

================================================================================
"""
