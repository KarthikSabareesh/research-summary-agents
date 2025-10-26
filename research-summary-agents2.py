import os
import re
from io import StringIO
from typing import Annotated, Literal, TypedDict, Optional
from datetime import datetime
from urllib.parse import urlparse
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

GUARDRAILS_ENABLED = True

class CustomGuard:
    @staticmethod
    def detect_pii(text: str) -> tuple[bool, list[str]]:
        pii_found = []
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            pii_found.append("EMAIL")
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            pii_found.append("PHONE")
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            pii_found.append("SSN")
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
            pii_found.append("CREDIT_CARD")
        if re.search(r'\b(sk-[a-zA-Z0-9]{32,}|pk-[a-zA-Z0-9]{32,})\b', text):
            pii_found.append("API_KEY")
        return len(pii_found) > 0, pii_found

    @staticmethod
    def detect_jailbreak(text: str) -> tuple[bool, str]:
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
        text_lower = text.lower()
        violence_keywords = [
            "genocide", "ethnic cleansing", "mass killing", "exterminate",
            "bomb", "weapon", "gun", "explosive", "grenade",
            "murder", "assassinate", "torture", "massacre",
            "slaughter", "lynch", "eradicate",
        ]
        for keyword in violence_keywords:
            if keyword in text_lower:
                return True, f"Violent content detected: '{keyword}'"
        hate_speech_patterns = [
            r"\b(scum|vermin|parasites|trash|garbage|filth)\b",
            r"(don't|do not|doesn't) deserve to (live|breathe|exist)",
            r"should (die|be killed|be eliminated|be exterminated|not exist)",
            r"(kill|eliminate|exterminate|eradicate) all (the )?\w+ (people|race|group)",
            r"genocide (all |the )?\w+ (people|race)",
            r"how to (kill|eliminate|harm|attack) (all |the )?\w+ (people|race|group)",
            r"\b(nazi|nazism|fascis[mt]|kkk|aryan nation)\w*\b",
            r"\b(white supremac|racial purity|master race)\w*\b",
            r"\b(hate|despise) (all |the )?\w+ (people|race|religion|group)",
            r"\w+ (people|race|group) (are|is) (inferior|subhuman|scum)",
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
        class ValidationResult:
            def __init__(self, passed: bool, error: str = None):
                self.validation_passed = passed
                self.error = error
                self.validated_output = text
        has_pii, pii_types = self.detect_pii(text)
        if has_pii:
            return ValidationResult(False, f"PII detected: {', '.join(pii_types)}")
        is_jailbreak, jb_msg = self.detect_jailbreak(text)
        if is_jailbreak:
            return ValidationResult(False, jb_msg)
        is_unsafe, unsafe_msg = self.detect_unsafe_content(text)
        if is_unsafe:
            return ValidationResult(False, unsafe_msg)
        return ValidationResult(True)

guard = CustomGuard()

class CitationManager:
    def __init__(self):
        self.citations = []
        self.url_index = {}

    def add_citation(self, url: str, title: str = "", author: str = "",
                     date: str = "", accessed_date: str = None) -> int:
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
            "domain": urlparse(url).netloc if url.startswith('http') else url
        }
        self.citations.append(citation)
        self.url_index[url] = citation_number
        return citation_number

    def _extract_title_from_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/').split('/')[-1]
            title = path.replace('-', ' ').replace('_', ' ').replace('.html', '').replace('.htm', '')
            return title.title() if title else parsed.netloc
        except:
            return url

    def format_citation(self, citation_number: int, style: str = "simple") -> str:
        if citation_number < 1 or citation_number > len(self.citations):
            return f"[Invalid citation: {citation_number}]"
        cit = self.citations[citation_number - 1]
        return f"[{citation_number}] {cit['title']} - {cit['domain']}\n    {cit['url']}"

    def get_all_citations(self, style: str = "simple") -> list[str]:
        return [self.format_citation(i+1, style) for i in range(len(self.citations))]

    def clear(self):
        self.citations = []
        self.url_index = {}

citation_manager = CitationManager()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)

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

existing = {idx.name for idx in pc.list_indexes().indexes}
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

semantic_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

bm25_documents = []
bm25_retriever = BM25Retriever.from_documents(
    bm25_documents if bm25_documents else [Document(page_content="initialization", metadata={"type": "init"})],
    k=3
)

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5],
)

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_search_data",
    "Hybrid search combining keyword (BM25) and semantic search to find relevant information from the knowledge base.",
    document_separator="\n\n---\n\n",
    response_format="content",
)

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
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

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

research_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
summary_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

class GuardrailsViolationError(Exception):
    pass

def apply_guardrails(text: str, context: str = "input") -> str:
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
    messages = state["messages"]
    user_query = messages[0].content if messages else ""

    global citation_manager
    print(f"[Research Agent] Clearing citation manager (had {len(citation_manager.citations)} citations)")
    citation_manager.clear()
    print(f"[Research Agent] Citation manager cleared (now has {len(citation_manager.citations)} citations)")

    try:
        safe_query = apply_guardrails(user_query, "input")
    except GuardrailsViolationError as e:
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

    need_web_search = True
    if kb_results and kb_results.strip():
        print("[Research Agent] Step 2: Evaluating knowledge base sufficiency...")
        query_terms = set(re.findall(r'\b\w{4,}\b', safe_query.lower()))
        kb_lower = kb_results.lower()
        matching_terms = [term for term in query_terms if term in kb_lower]
        match_ratio = len(matching_terms) / max(len(query_terms), 1)
        print(f"[Research Agent]   Keyword check: {len(matching_terms)}/{len(query_terms)} terms matched ({match_ratio:.1%})")

        if match_ratio < 0.75:
            print("[Research Agent] ○ Low keyword overlap - KB not relevant, will perform web search")
            need_web_search = True
        else:
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

    web_results = ""
    documents_to_store = []

    if need_web_search and TAVILY_ENABLED and tavily_tool:
        print("[Research Agent] Step 3: Performing web search via Tavily...")
        try:
            tavily_results = tavily_tool.invoke({"query": safe_query})
            web_results_list = []
            print(f"[Research Agent] ✓ Retrieved web search results")

            try:
                if isinstance(tavily_results, list):
                    for idx, item in enumerate(tavily_results):
                        if isinstance(item, dict):
                            content = item.get('content', '') or item.get('snippet', '')
                            url = item.get('url', 'unknown')
                            title = item.get('title', '')

                            if content and url != 'unknown':
                                citation_num = citation_manager.add_citation(
                                    url=url,
                                    title=title
                                )
                                print(f"[Research Agent] Added citation #{citation_num}: {title[:50]}... (Total: {len(citation_manager.citations)})")

                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": url,
                                        "query": safe_query,
                                        "type": "tavily_search",
                                        "title": title,
                                        "citation_number": citation_num
                                    }
                                )
                                documents_to_store.append(doc)

                                web_results_list.append(
                                    f"[Source {citation_num}]\n"
                                    f"URL: {url}\n"
                                    f"Content: {content}\n"
                                )

                    web_results = "\n---\n".join(web_results_list)

                elif isinstance(tavily_results, str) and tavily_results.strip():
                    doc = Document(
                        page_content=tavily_results,
                        metadata={"source": "tavily", "query": safe_query, "type": "tavily_search"}
                    )
                    documents_to_store.append(doc)
                    web_results = tavily_results

            except Exception as e:
                print(f"[Research Agent] Warning: Could not parse Tavily results: {e}")
                web_results = str(tavily_results)

        except Exception as e:
            print(f"[Research Agent] ○ Web search failed: {e}")
            web_results = ""
    elif need_web_search and not TAVILY_ENABLED:
        print("[Research Agent] ⚠ Web search needed but Tavily is not enabled")

    if documents_to_store:
        try:
            doc_chunks = text_splitter.split_documents(documents_to_store)
            vectorstore.add_documents(doc_chunks)

            global bm25_retriever
            existing_docs = getattr(bm25_retriever, 'docs', [])
            existing_docs = [doc for doc in existing_docs if doc.metadata.get("type") != "init"]
            all_docs = existing_docs + doc_chunks
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=3)

            global retriever
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5],
            )

            print(f"[Research Agent] ✓ Stored {len(doc_chunks)} chunks in hybrid knowledge base (BM25 + semantic)")
        except Exception as e:
            print(f"[Research Agent] Warning: Could not store documents: {e}")

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

    try:
        safe_findings = apply_guardrails(research_findings, "output")
    except GuardrailsViolationError as e:
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
                    "citations_count": len(citation_manager.citations)
                }
            )
        ]
    }

from pydantic import BaseModel, Field

class ExecutiveSummary(BaseModel):
    title: str = Field(description="Brief title for the research")
    key_findings: list[str] = Field(description="3-5 key findings from the research")
    summary: str = Field(description="Concise executive summary (2-3 paragraphs)")
    sources: list[str] = Field(description="List of sources cited")
    confidence_level: str = Field(description="Confidence level: high, medium, or low")

def summary_agent(state: MessagesState):
    messages = state["messages"]
    original_query = messages[0].content if messages else ""
    research_findings = messages[-1].content if len(messages) > 1 else ""

    citations_count = 0
    if len(messages) > 1 and hasattr(messages[-1], 'additional_kwargs'):
        additional_kwargs = messages[-1].additional_kwargs
        citations_count = additional_kwargs.get('citations_count', 0)

        if additional_kwargs.get('blocked', False):
            return {
                "messages": [
                    AIMessage(
                        content=research_findings,
                        name="summary_agent",
                        additional_kwargs={"agent": "summary", "blocked": True, "passthrough": True}
                    )
                ]
            }

    system_prompt = """You are a Summary Agent specialized in creating executive summaries.

Your task:
1. Analyze the research findings provided by the Research Agent
2. Extract key insights and important information
3. Create a concise, well-structured executive summary
4. Organize information in a clear, actionable format
5. Cite sources with [Source N] notation

Format your summary for business/executive audience with:
- Clear title
- Key findings (bullet points with source citations)
- Executive summary (2-3 paragraphs)
- Source citations
- Confidence assessment"""

    summary_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Original Query: {original_query}

Research Findings:
{research_findings}

Please create an executive summary of these research findings.""")
    ]

    model_with_structure = summary_model.with_structured_output(ExecutiveSummary)
    summary_result = model_with_structure.invoke(summary_messages)

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

    try:
        safe_summary = apply_guardrails(formatted_summary, "final_output")
    except GuardrailsViolationError as e:
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

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

workflow.add_node("research_agent", research_agent)
workflow.add_node("summary_agent", summary_agent)

workflow.add_edge(START, "research_agent")
workflow.add_edge("research_agent", "summary_agent")
workflow.add_edge("summary_agent", END)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

print("\nMulti-Agent Research Assistant initialized successfully!")
print("- Research Agent: Web search (Tavily) + Hybrid RAG (BM25 + Semantic)")
print("- Summary Agent: Executive summary generation")
print(f"- Guardrails: {'Enabled' if GUARDRAILS_ENABLED else 'Disabled'}")
print(f"- Tavily Search: {'Enabled' if TAVILY_ENABLED else 'Disabled'}")
print("- Retrieval: Hybrid search with 50% BM25 (keyword) + 50% semantic")

from langchain_core.messages import BaseMessage

def run_research_query(query: str, thread_id: str = "demo-1", verbose: bool = True):
    print(f"\n{'='*80}")
    print(f"Research Query: {query}")
    print(f"{'='*80}\n")

    config = {"configurable": {"thread_id": thread_id}}

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    ):
        for node_name, update in chunk.items():
            msg = update["messages"][-1]

            if verbose:
                print(f"\n[{node_name.upper()}]")
                print("-" * 80)

            if isinstance(msg, BaseMessage):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = getattr(msg, "content", "")

            if verbose and content:
                if node_name == "research_agent":
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(content)

    print(f"\n{'='*80}")
    print("Research complete!")
    print(f"{'='*80}\n")
