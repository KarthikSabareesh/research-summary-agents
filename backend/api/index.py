# backend/api/index.py
"""
Vercel's Python runtime looks for a module-level `app` (WSGI) inside files
under the `api/` directory. This shim simply exposes your existing Flask app
defined in backend/app.py, without changing your repo structure.
"""

# First try to import a module-level Flask instance named `app`
try:
    from app import app as _flask_app  # backend/app.py must export: app = Flask(__name__)
except Exception:
    # If you use an application factory pattern (create_app), fall back to that.
    # Adjust the import if your factory has a different name or signature.
    from app import create_app  # backend/app.py must define create_app()
    _flask_app = create_app()

# Vercel requires this exact name:
app = _flask_app

# Optional: allow running this file directly for local testing (not used on Vercel)
if __name__ == "__main__":
    # Bind to 0.0.0.0 so you can test via localhost; port can be overridden by $PORT.
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
