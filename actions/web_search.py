import subprocess
from urllib.parse import quote_plus


CHROME_PATH = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe"
)

CHROME_PROFILE = "Default"


def open_chrome_url(url):
    subprocess.Popen([
        CHROME_PATH,
        f"--profile-directory={CHROME_PROFILE}",
        "--new-tab",
        url
    ])


def search_google(query):
    query = query.strip()

    if not query:
        return "Please tell me what to search for."

    encoded_query = quote_plus(query)

    url = (
        "https://www.google.com/search"
        f"?q={encoded_query}"
    )

    open_chrome_url(url)

    return f"Searching Google for {query}."


def search_youtube(query):
    query = query.strip()

    if not query:
        return "Please tell me what to search for."

    encoded_query = quote_plus(query)

    url = (
        "https://www.youtube.com/results"
        f"?search_query={encoded_query}"
    )

    open_chrome_url(url)

    return f"Searching YouTube for {query}."


def search_web(query, engine="google"):
    engine = engine.lower().strip()

    if engine in {"youtube", "yt"}:
        return search_youtube(query)

    return search_google(query)


if __name__ == "__main__":
    print("=========================")
    print("    WEB SEARCH TEST")
    print("=========================")

    selected_engine = input(
        "Engine (Google/YouTube): "
    ).strip()

    search_query = input(
        "What should I search for?: "
    ).strip()

    result = search_web(
        search_query,
        selected_engine
    )

    print(result)