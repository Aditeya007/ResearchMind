import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from urllib.parse import urlparse


def load_url(url: str) -> List[Dict[str, Any]]:
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch URL: {url}\nReason: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

   
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    
    text = soup.get_text(separator="\n")
    text = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )

    if not text:
        raise ValueError(f"No text content found at: {url}")

    domain = urlparse(url).netloc

    documents = [{
        "text": text,
        "metadata": {
            "source": domain,
            "source_type": "web",
            "url": url,
            "title": soup.title.string.strip() if soup.title else domain,
        },
    }]

    print(f"[web_loader] Loaded '{documents[0]['metadata']['title']}' from {domain}")
    return documents


def load_urls(urls: List[str]) -> List[Dict[str, Any]]:
    
    all_docs = []

    for url in urls:
        try:
            docs = load_url(url)
            all_docs.extend(docs)
        except Exception as e:
            print(f"[web_loader] Skipping {url} — {e}")

    return all_docs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Error: No URL provided.")
        sys.exit(1)
    url = sys.argv[1] 
    docs = load_url(url)
    print(f"\nTitle   : {docs[0]['metadata']['title']}")
    print(f"Source  : {docs[0]['metadata']['source']}")
    print(f"Preview :\n{docs[0]['text'][:300]}")