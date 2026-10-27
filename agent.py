
import os
import requests
import subprocess

def read_page(url: str) -> dict:
    result = subprocess.run(
        [
            "obscura",
            "fetch",
            url,
            "--dump",
            "markdown",
            "--timeout",
            "30",
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )

    if result.returncode != 0:
        return {
            "success": False,
            "url": url,
            "error": result.stderr[:2000],
        }

    # return {
    #     "success": True,
    #     "url": url,
    #     "content": result.stdout[:8000],
    #     "truncated": len(result.stdout) > 8000,
    # }

    content = result.stdout.strip()

    # Locate the beginning of the .NET 10 article
    start_marker = "# What's new in .NET 10"
    start = content.find(start_marker)

    if start >= 0:
        content = content[start:]

    # Limit the content sent to the LLM
    content = content[:12000]

    return {
        "success": True,
        "url": url,
        "content": content,
        "page_characters": len(content),
        "contains_dotnet_10": ".NET 10" in content
    }

SEARXNG_URL = os.getenv(
    "SEARXNG_URL",
    "http://localhost:8080"
).rstrip("/")


def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web through a local SearXNG instance."""

    query = query.strip()

    if not query:
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": "Search query cannot be empty."
        }

    max_results = max(1, min(int(max_results), 10))

    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
                "safesearch": 1
            },
            headers={
                "User-Agent": "LocalResearchAgent/1.0"
            },
            timeout=(5, 25)
        )

        if response.status_code == 403:
            return {
                "success": False,
                "query": query,
                "results": [],
                "error": (
                    "SearXNG returned HTTP 403. "
                    "Check that JSON output is enabled "
                    "in settings.yml."
                )
            }

        response.raise_for_status()
        data = response.json()

        raw_results = data.get("results", [])
        results = []

        for item in raw_results:
            url = item.get("url", "").strip()

            # Ignore malformed or unsupported result URLs.
            if not url.startswith(("https://", "http://")):
                continue

            results.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("content", ""),
                "engine": item.get("engine", "")
            })

            if len(results) >= max_results:
                break

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
            "error": None if results else
                "No usable results were returned."
        }

    except requests.Timeout:
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": "SearXNG search timed out."
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": f"Could not reach SearXNG: {exc}"
        }

    except ValueError:
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": (
                "SearXNG did not return valid JSON. "
                "Check the API endpoint and JSON settings."
            )
        }
