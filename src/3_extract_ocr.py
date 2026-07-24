import re
import json
import csv
from pathlib import Path
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RECEIPTS_DIR = Path("data/receipts")
LABELS_DIR = Path("data/labels")
OUTPUT_CSV = LABELS_DIR / "extracted_data.csv"

DATE_PATTERN = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")

# broader set of labels real receipts use near the total, checked in this order
TOTAL_LABEL_PATTERNS = [
    re.compile(r"grand\s*total[^\d]{0,15}(\d+[.,]\d{2})", re.IGNORECASE),
    re.compile(r"total\s*amount[^\d]{0,15}(\d+[.,]\d{2})", re.IGNORECASE),
    re.compile(r"amount\s*due[^\d]{0,15}(\d+[.,]\d{2})", re.IGNORECASE),
    re.compile(r"\btotal\b[^\d]{0,15}(\d+[.,]\d{2})", re.IGNORECASE),
    re.compile(r"\bamt\b[^\d]{0,15}(\d+[.,]\d{2})", re.IGNORECASE),
]

# fallback: if no labeled total found, grab every price-like number and take the largest,
# since the total is usually the biggest number on a receipt
ANY_PRICE_PATTERN = re.compile(r"(\d+[.,]\d{2})")


def extract_fields(text):
    date_match = DATE_PATTERN.search(text)

    total_value = None
    for pattern in TOTAL_LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            total_value = match.group(1)
            break

    if total_value is None:
        all_prices = ANY_PRICE_PATTERN.findall(text)
        if all_prices:
            cleaned = [p.replace(",", ".") for p in all_prices]
            total_value = max(cleaned, key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else 0)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    vendor = lines[0] if lines else ""

    return {
        "vendor_ocr": vendor,
        "date_ocr": date_match.group(0) if date_match else "",
        "total_ocr": total_value or "",
    }


def normalize_amount(value):
    # strips currency symbols/letters, keeps digits and one decimal point, returns a float or None
    if not value:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", "."))
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def main():
    rows = []
    correct_totals, checked = 0, 0

    for image_path in sorted(RECEIPTS_DIR.glob("*.jpg")):
        text = pytesseract.image_to_string(Image.open(image_path))
        extracted = extract_fields(text)
        extracted["image_filename"] = image_path.name

        label_file = LABELS_DIR / (image_path.stem + ".txt")
        if label_file.exists():
            try:
                truth = json.loads(label_file.read_text(encoding="utf-8", errors="ignore"))
                truth_amount = normalize_amount(truth.get("total", ""))
                ocr_amount = normalize_amount(extracted["total_ocr"])
                if truth_amount is not None:
                    checked += 1
                    # allow a tiny tolerance for rounding differences, still counts as a real match
                    if ocr_amount is not None and abs(truth_amount - ocr_amount) < 0.01:
                        correct_totals += 1
            except json.JSONDecodeError:
                pass

        rows.append(extracted)
        print(f"{image_path.name}: vendor={extracted['vendor_ocr'][:30]}, total={extracted['total_ocr']}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "vendor_ocr", "date_ocr", "total_ocr"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtracted data from {len(rows)} receipts.")
    if checked:
        print(f"Total field OCR accuracy vs ground truth: {correct_totals / checked:.1%}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()