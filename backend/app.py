from dotenv import load_dotenv
import os
load_dotenv()  # Load .env file from project root


print("\n" + "="*80)
print("ENVIRONMENT VARIABLES CHECK")
print("="*80)
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_API_KEY: {os.getenv('LANGCHAIN_API_KEY', 'NOT SET')[:20]}..." if os.getenv('LANGCHAIN_API_KEY') else "NOT SET")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print("="*80 + "\n")

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import json
from datetime import datetime
import traceback


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

# Global variables for the agent system
graph = None
citation_manager = None
initialized = False
initialization_error = None

def initialize_agent_system():
    """Initialize the research agent system."""
    global graph, citation_manager, initialized, initialization_error

    try:
        # Import the entire research agent module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "research_agent",
            os.path.join(os.path.dirname(__file__), '..', 'research-summary-agents2.py')
        )
        research_agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(research_agent_module)

        # Get the compiled graph and citation manager
        graph = research_agent_module.graph
        citation_manager = research_agent_module.citation_manager

        initialized = True
        print("✓ Research agent system initialized successfully")
        return True

    except Exception as e:
        initialization_error = str(e)
        print(f"✗ Failed to initialize research agent system: {e}")
        traceback.print_exc()
        return False

# Initialize on startup
initialize_agent_system()


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        'status': 'online' if initialized else 'error',
        'initialized': initialized,
        'error': initialization_error,
        'timestamp': datetime.now().isoformat(),
        'features': {
            'hybrid_search': True,
            'citation_management': True,
            'guardrails': True
        } if initialized else {}
    })


@app.route('/api/query', methods=['POST'])
def query():
    """
    Submit a research query and get executive summary.
    """
    if not initialized:
        return jsonify({
            'success': False,
            'error': 'System not initialized',
            'details': initialization_error
        }), 500

    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: query'
            }), 400

        user_query = data['query']
        thread_id = data.get('thread_id', f'session-{datetime.now().timestamp()}')

        print(f"\n[API] Processing query: {user_query[:100]}...")
        print(f"[API] Thread ID: {thread_id}")

        # Run the research query through the agent graph
        config = {"configurable": {"thread_id": thread_id}}

        # Collect all results
        results = []
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": user_query}]},
            config=config
        ):
            results.append(chunk)

        # Extract the final summary from the last chunk (summary_agent)
        final_summary = None
        metadata = {}

        for chunk in results:
            for node_name, update in chunk.items():
                if node_name == "summary_agent":
                    msg = update["messages"][-1]
                    final_summary = msg.content

                    # Extract metadata if available
                    if hasattr(msg, 'additional_kwargs'):
                        metadata = {
                            'agent': msg.additional_kwargs.get('agent'),
                            'blocked': msg.additional_kwargs.get('blocked', False)
                        }

                elif node_name == "research_agent":
                    msg = update["messages"][-1]
                    if hasattr(msg, 'additional_kwargs'):
                        metadata.update({
                            'citations_count': msg.additional_kwargs.get('citations_count', 0),
                            'used_kb': msg.additional_kwargs.get('used_kb', False),
                            'used_web_search': msg.additional_kwargs.get('used_web_search', False)
                        })

        # Get citations
        citations = []
        citations_count = metadata.get('citations_count', 0)
        print(f"\n[API] ===== CITATIONS DEBUG =====")
        print(f"[API] Citations count from metadata: {citations_count}")
        print(f"[API] Citation manager exists: {citation_manager is not None}")

        if citation_manager:
            print(f"[API] Citation manager has {len(citation_manager.citations)} citations in memory")
            print(f"[API] Citation manager ID: {id(citation_manager)}")

            # Always try to get citations if citation_manager has any
            if len(citation_manager.citations) > 0:
                citations = citation_manager.get_all_citations(style='simple')
                print(f"[API] Retrieved {len(citations)} formatted citations")
                for i, cit in enumerate(citations[:3]):  # Print first 3
                    print(f"[API] Citation {i+1}: {cit[:100]}...")
            else:
                print(f"[API] Citation manager is empty!")
        else:
            print(f"[API] Citation manager is None!")

        print(f"[API] ===== END CITATIONS DEBUG =====\n")

        response_data = {
            'success': True,
            'query': user_query,
            'summary': final_summary,
            'metadata': metadata,
            'citations': citations,
            'timestamp': datetime.now().isoformat()
        }

        print(f"[API] ✓ Query processed successfully")
        print(f"[API] Response will include {len(citations)} citations")

        return jsonify(response_data)

    except Exception as e:
        print(f"[API] ✗ Error processing query: {e}")
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Research Summary Agents API Server")
    print("="*80)
    print(f"Initialized: {initialized}")
    if initialization_error:
        print(f"Error: {initialization_error}")
    print("\nEndpoints:")
    print("  GET  /api/status       - System status")
    print("  GET  /api/health       - Health check")
    print("  POST /api/query        - Submit research query")
    print("\nStarting server on http://localhost:5000")
    print("="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
