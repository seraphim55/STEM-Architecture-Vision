import io
import json
import os
import time
import urllib.parse
import urllib.request
from PIL import Image

DATASET_DIR = "dataset"
TARGET_COUNT = 120
IMG_SIZE = (256, 256)

os.makedirs(DATASET_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "STEM-Architecture-Project/1.0 (Educational KNU Student Project)"
}

SEARCH_QUERIES = [
    "filetype:bitmap architecture building facade",
    "filetype:bitmap historic bridge architecture",
    "filetype:bitmap palace exterior architecture",
    "filetype:bitmap modern building architecture",
    "filetype:bitmap cathedral tower architecture"
]


def fetch_image_urls(query: str, limit: int = 50) -> list:
    """Отримує прямі посилання на зображення через відкритий API Wikimedia Commons."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "400"
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)

    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        title = page.get("title", "")
        info = page.get("imageinfo", [{}])[0]
        img_url = info.get("thumburl") or info.get("url")
        if img_url:
            results.append((title, img_url))
    return results


def main():
    existing = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".png")]
    saved_count = len(existing)
    seen_titles = set()

    print(f"Поточна кількість зображень у '{DATASET_DIR}': {saved_count}/{TARGET_COUNT}")

    for query in SEARCH_QUERIES:
        if saved_count >= TARGET_COUNT:
            break

        print(f"Пошук зображень: '{query}'...")
        try:
            candidates = fetch_image_urls(query, limit=50)
        except Exception as e:
            print(f"Помилка запиту: {e}")
            continue

        for title, img_url in candidates:
            if saved_count >= TARGET_COUNT:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)

            try:
                req = urllib.request.Request(img_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    img_bytes = resp.read()

                with Image.open(io.BytesIO(img_bytes)) as img:
                    img_rgb = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    filename = f"arch_{saved_count + 1:03d}.png"
                    img_rgb.save(os.path.join(DATASET_DIR, filename), "PNG")

                saved_count += 1
                if saved_count % 20 == 0 or saved_count == TARGET_COUNT:
                    print(f"Завантажено: {saved_count}/{TARGET_COUNT}")

                time.sleep(0.15)
            except Exception:
                continue

    print(f"Готово! У папці '{DATASET_DIR}' зібрано {saved_count} зображень.")


if __name__ == "__main__":
    main()