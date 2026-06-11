# The Problem Statement: Why AURA?

> **[← Back to README](../README.md)**

---

## 🛑 The Core Problem: The Earnings Season Information Overload
Every quarter, public companies release massive, 30+ page earnings call transcripts. For financial analysts, portfolio managers, and individual investors, extracting actionable intelligence from these unstructured walls of text is an exhausting, manual process. 

Finding exactly what a CEO said about "supply chain headwinds," tracking quarter-over-quarter gross margin trends, or comparing the AI infrastructure investments of Microsoft vs. Nvidia requires reading dozens of documents simultaneously. 

### Why Software Giants & Traditional AI Struggle
When firms try to automate this using standard AI, they hit catastrophic failures:
1. **Hallucination of Financials**: Standard LLMs invent revenue numbers or mix up Q1 and Q2 data.
2. **Entity Starvation**: When asked to *"Compare Apple, Microsoft, and Nvidia's AI strategies,"* traditional Retrieval-Augmented Generation (RAG) systems fetch 10 documents about Microsoft (because they matched the keywords best) and 0 about Apple, resulting in heavily biased or incomplete analyses.
3. **Loss of Source Truth**: Investors cannot trust an AI summary unless they can instantly click and verify the exact sentence the CEO said. Standard RAG loses citation mapping, especially when formatting data into comparative tables.

---

## 💡 The AURA Solution: A Paradigm Shift in Financial RAG
AURA is built to solve exactly these enterprise-level challenges using a **Multi-Agent RAG Architecture**:

- **Zero-Hallucination Quantitative Data**: Instead of guessing numbers from text, AURA employs a dedicated Data Analyst Agent that pulls exact KPIs (Revenue, EPS, Guidance) directly from a structured SQLite database.
- **Fair Multi-Entity RAG**: AURA's retrieval engine uses a custom 3-layer Quota Allocation algorithm. If you ask about 3 companies, the engine guarantees equal context window representation for Apple, Microsoft, and Nvidia—completely eliminating entity starvation.
- **Hybrid Cognitive Search**: By fusing dense semantic search (ChromaDB vectors) with exact lexical matching (BM25) and applying a deep Cross-Encoder reranker, AURA understands the difference between a casual mention of "margins" and a CFO's forward-looking margin guidance.

---

## 📊 The Dataset
AURA is powered by real-world financial data. The base transcripts were sourced from the Kaggle dataset:
**[Earning Call Transcripts (2023-2024)](https://www.kaggle.com/datasets/ramssvimala/earning-call-transcripts)**

### Data Preparation
To prepare the dataset for AURA's ingestion pipeline, the raw data was filtered down to three major tech giants (Apple, Microsoft, Nvidia) and restructured using the following automation script to preserve a clean directory hierarchy:

```powershell
# Create target folder
New-Item -ItemType Directory -Force -Path raw_data

# Create company folders
$companies = @("Apple", "Microsoft", "Nvidia")

foreach ($company in $companies) {

    New-Item -ItemType Directory -Force `
        -Path "raw_data\Earning_Call_Transcripts\cleaned_ECTs_dataset\$company"

    Copy-Item `
        "Earning_Call_Transcripts\cleaned_ECTs_dataset\$company\2023_*" `
        "raw_data\Earning_Call_Transcripts\cleaned_ECTs_dataset\$company\" `
        -Force

    Copy-Item `
        "Earning_Call_Transcripts\cleaned_ECTs_dataset\$company\2024_*" `
        "raw_data\Earning_Call_Transcripts\cleaned_ECTs_dataset\$company\" `
        -Force
}
```

This restructuring results in the exact hierarchy the AURA ingestion engine requires:
```text
raw_data
└── Earning_Call_Transcripts
    └── cleaned_ECTs_dataset
        ├── Apple
        │   ├── 2023_Q1_aapl_processed.txt
        │   ├── ...
        │   └── 2024_Q3_aapl_processed.txt
        │
        ├── Microsoft
        │   ├── 2023_Q1_msft_processed.txt
        │   ├── ...
        │   └── 2024_Q4_msft_processed.txt
        │
        └── Nvidia
            ├── 2023_Q1_nvda_processed.txt
            ├── ...
            └── 2024_Q4_nvda_processed.txt
```
