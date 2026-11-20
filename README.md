# Smart Expense Intelligence

From a photo of a receipt to a business insight, in one pipeline.

This is an end to end project that takes scanned receipt images, classifies each one into an expense category using a trained image classifier, extracts vendor, date, and total using OCR, stores the results in a database, forecasts future spend, and lets you ask plain English questions about your spending through an AI chatbot grounded in the real data.

## Live demo

The live site is a dashboard and a spending chatbot over an **anonymised demo copy** of the data (vendor names replaced with "Vendor 001" and so on). The receipt classifier, OCR and forecasting ran offline to build the database. The live site shows their results, and it cannot classify a new receipt.

The chatbot is SQL-grounded Q&A, not vector search. The code computes the totals, and a language model explains them in plain English. Each visitor is limited to 8 questions an hour and the demo stops at 300 a day to protect a free AI quota. The counters live in memory, so they reset whenever the free instance restarts, and the per-visitor limit is a courtesy, not a security feature.

It is hosted on a free tier, so the first visit after a quiet spell can take about a minute to wake up.


## About the data

The receipts come from the ICDAR 2019 SROIE scanned-receipt competition dataset. I could not confirm its licence terms, so the raw receipts, their annotations and my OCR output are not included in this repo. The live demo uses an anonymised copy of the transactions. To reproduce the pipeline, download the dataset from its original source.

## How it works

```
Receipt image
   |
   |-- 1. Category labeling (keyword rules + Gemini AI, weak supervision)
   |-- 2. Image classifier (PyTorch, transfer learning on ResNet18)
   |-- 3. OCR extraction (Tesseract: vendor, date, total)
   |-- 4. SQLite database (cleaned, validated transactions)
   |-- 5. Spend forecasting (Prophet, 3 months ahead per category)
   |-- 6. RAG chatbot (Gemini, answers grounded in real data)
   |
   |-- Flask web app + Power BI dashboard
```

## Tech stack

- **Python**, pandas, scikit-learn
- **PyTorch**, transfer learning on ResNet18 for image classification
- **Tesseract OCR** via pytesseract
- **SQLite** for storage
- **Prophet** for time series forecasting
- **Google Gemini API** for AI labeling and the RAG chatbot
- **Flask** for the web app
- **Power BI** for the dashboard (.pbix included, PDF export included)

## Dataset

[SROIE 2019](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2), a public dataset of real scanned receipts with ground truth annotations, originally built for OCR research. Not included in this repo, download it from Kaggle and place images in `data/receipts/` and annotation files in `data/labels/`.

## How to run it

1. Clone the repo and set up the environment:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Install Tesseract OCR for Windows from https://github.com/UB-Mannheim/tesseract/wiki and update the path at the top of `src/3_extract_ocr.py` if needed.

3. Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

4. Run the pipeline in order:

```
python src/1_label_receipts.py
python src/2_train_classifier.py
python src/3_extract_ocr.py
python src/4_build_database.py
python src/5_forecast.py
```

5. Start the web app:

```
python app.py
```

Then open http://127.0.0.1:5000

## Honest notes on results

This project prioritizes a real, working end to end pipeline over inflated metrics. The classifier and OCR results reflect the genuine challenges of a small, imbalanced, real world dataset, and the writeup documents the problems hit along the way (class imbalance, OCR misreads producing impossible dates, API rate limits) and how each was solved. See `PROJECT_WRITEUP.md` for the full story.


## Author

Ruchira Surendra, Data Analyst, Melbourne
[LinkedIn](https://linkedin.com/in/ruchira-surendra) 