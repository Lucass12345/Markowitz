"""Server-side HTTP credentials; never send keys to the Streamlit browser."""
import os


def api_headers():
    value=os.getenv('MARKOWITZ_API_KEY') or os.getenv('MARKOWITZ_READ_API_KEY')
    return {'X-API-Key':value} if value else {}



def api_headers_for(url, trusted_bases):
    from urllib.parse import urlsplit
    destination=urlsplit(url)
    if any((destination.scheme,destination.netloc)==(urlsplit(base).scheme,urlsplit(base).netloc) for base in trusted_bases):
        return api_headers()
    return {}
