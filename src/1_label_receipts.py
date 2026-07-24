import os
import json
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

LABELS_DIR = Path("data/labels")
RECEIPTS_DIR = Path("data/receipts")
OUTPUT_CSV = LABELS_DIR / "category_labels.csv"

CATEGORIES = ["Meals", "Travel", "Office Supplies", "Utilities", "Retail"]

RULES = {
    "Meals": ["cafe", "coffee", "restaurant", "restoran", "kedai kopi", "kopitiam",
              "food", "kitchen", "bakery", "diner", "pizza", "warung", "gerai"],
    "Travel": ["petrol", "fuel", "shell", "taxi", "uber", "parking", "airline", "hotel"],
    "Office Supplies": ["stationery", "office", "print", "paper", "staples", "bookshop", "book co"],
    "Utilities": ["electric", "water", "gas", "telco", "telecom", "internet"],
    "Retail": ["hardware", "trading", "enterprise", "farmasi", "pharmacy", "gift",
               "deco", "mart", "supermarket", "stores"],
}


def rule_based_category(vendor_name):
    name = vendor_name.lower()
    for category, keywords in RULES.items():
        if any(kw in name for kw in keywords):
            return category
    return None


def llm_category(vendor_name):
    prompt = (
        f"Classify this business vendor name into exactly one of these categories: "
        f"{', '.join(CATEGORIES)}.\nVendor name: \"{vendor_name}\"\n"
        f"Reply with only the category name, nothing else."
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
            answer = response.text.strip()
            time.sleep(13)
            return answer if answer in CATEGORIES else "Retail"
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print("Rate limit hit, waiting 20 seconds before retrying...")
                time.sleep(20)
            else:
                raise
    return "Retail"


def main():
    label_files = sorted(LABELS_DIR.glob("*.txt"))
    vendor_cache = {}
    rule_hits, llm_hits = 0, 0
    processed = 0

    # write the header once, then append one row at a time as we go,
    # so progress is saved immediately and never lost if we stop partway
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "vendor", "category"])
        writer.writeheader()

    for label_file in label_files:
        try:
            data = json.loads(label_file.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue

        vendor = data.get("company", "").strip()
        if not vendor:
            continue

        image_name = label_file.stem + ".jpg"

        if vendor in vendor_cache:
            category = vendor_cache[vendor]
        else:
            category = rule_based_category(vendor)
            if category:
                rule_hits += 1
            else:
                category = llm_category(vendor)
                llm_hits += 1
            vendor_cache[vendor] = category

        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_filename", "vendor", "category"])
            writer.writerow({"image_filename": image_name, "vendor": vendor, "category": category})

        processed += 1
        print(f"[{processed}/{len(label_files)}] {vendor} -> {category}")

    print(f"\nDone. Labeled {processed} receipts. Rule matches: {rule_hits}, AI matches: {llm_hits}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()