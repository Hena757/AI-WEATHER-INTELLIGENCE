"""WSGI entry point for production deployment.

Run with gunicorn:
    gunicorn wsgi:app

Or with waitress:
    waitress-serve --port=5000 wsgi:app
"""

from api.app import app

if __name__ == "__main__":
    app.run()