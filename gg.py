from bing_image_downloader import downloader
import os

# ==============================
# CONFIG
# ==============================
OUTPUT_DIR = "dataset/other"
IMAGES_PER_QUERY = 40   # 15 queries × 40 ≈ 600 images

QUERIES = [

    # Everyday objects
    "random everyday objects",
    "household items on table",
    "kitchen utensils close up",
    "office desk items",
    "tools and hardware objects",
    "plastic objects",
    "metal objects",
    "wooden objects",
    "fabric textures",

    # Indoor scenes
    "indoor scene",
    "living room interior",
    "bedroom interior",
    "kitchen interior",
    "classroom interior",
    "shopping mall interior",
    "restaurant interior",
    "warehouse interior",
    "hospital room interior",

    # Outdoor scenes
    "outdoor scene",
    "urban street view",
    "village road",
    "mountain landscape",
    "river bank",
    "agricultural field",
    "city skyline",
    "construction site",
    "park with people",

    # Nature & background variety
    "grass field",
    "soil ground texture",
    "tree bark texture",
    "cloudy sky",
    "sunset landscape",
    "rainy weather scene",
    "foggy morning",

    # Human presence (for realism diversity)
    "people walking",
    "crowd in market",
    "farmer working in field",
    "person using smartphone",
    "children playing outside",

    # Close-up / macro diversity
    "macro photography texture",
    "blurred background bokeh",
    "low light indoor photo",
    "overexposed image example",
    "motion blur image"
]

# ==============================
# DOWNLOAD
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for query in QUERIES:
    print(f"[INFO] Downloading images for: {query}")
    downloader.download(
        query=query,
        limit=IMAGES_PER_QUERY,
        output_dir=OUTPUT_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("\n[INFO] Download complete.")
print("[INFO] Total expected images ≈", len(QUERIES) * IMAGES_PER_QUERY)
