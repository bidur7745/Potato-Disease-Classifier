from bing_image_downloader import downloader
import os

# ==============================
# CONFIG
# ==============================
OUTPUT_DIR = "dataset/other"
IMAGES_PER_QUERY = 40   # 15 queries × 40 ≈ 600 images

QUERIES = [

    
  

    # Nature & background variety
    "grass field",
    "soil ground texture",
    "tree bark texture",
    "cloudy sky",
    "sunset landscape",
    "rainy weather scene",
    "foggy morning",

      # Close-up / macro diversity
    "macro photography texture",
    "blurred background bokeh",
    "low light indoor photo",
    "overexposed image example",
    "motion blur image",


    # Human presence (for realism diversity)
    "people walking",
    "crowd in market",
    "farmer working in field",
    "person using smartphone",
    "children playing outside",

  
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
