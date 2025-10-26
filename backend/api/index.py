# backend/api/index.py
"""
Expose the Flask WSGI `app` from backend/app.py to Vercel's Python runtime.
Vercel looks for a top-level variable named `app` in files under /api.
"""

import os
import sys

# Ensure we can import from the parent directory (backend/)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import your Flask app instance from backend/app.py
from app import app  # expects: app = Flask(__name__)

# Optional: allow running this file directly for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
