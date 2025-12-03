# Local Information Retrieval System

A complete, local search engine built in Python using the **BM25 algorithm**. This system ingests news articles from a structured CSV file, processes the text using Natural Language Processing (NLP) techniques, and allows users to perform keyword-based searches via the command line.

## 📌 Features

- **Data Source:** Reads directly from a `data.csv` file.
- **Algorithm:** Uses **BM25Okapi** (Best Matching 25) for ranking, which offers superior relevance compared to standard TF-IDF.
- **Preprocessing:** Implements a full NLP pipeline including normalization, tokenization, stop-word removal, and stemming.
- **Efficiency:** Runs entirely locally on your machine with no cloud dependencies.
- **Real-time Results:** Provides instant search results with relevance scores and content previews.

## 🛠️ Prerequisites

Before running the system, ensure you have the following installed:

- **Python 3.x**
- The following Python libraries:
  - `rank_bm25` (For the ranking algorithm)
  - `nltk` (For text processing)

## 🚀 Installation & Setup

### 1. Clone or Download the Repository

Download the project files to a folder on your computer.

### 2. Install Dependencies

Open your terminal or command prompt and run the following command to install the required libraries:

````bash
pip install rank-bm25 nltk
````
### 3. Prepare the Data
Ensure you have your dataset file named **`data.csv`** in the same directory as the script.

The CSV file must contain at least two columns:
* **Heading:** Used as the document title.
* **Article:** Used as the content to be indexed.

## 💻 How to Run the System

1.  Open your terminal or command prompt.
2.  Navigate to the project directory using the `cd` command:
    ```bash
    cd path/to/your/project
    ```
3.  Run the main Python script:
    ```bash
    python IR_system.py
    ```
4.  **Note:** On the first run, the system will automatically download necessary NLTK data (like stop words). This happens only once.

## 🔍 Usage Guide

Once the system is running, you will see a prompt. Follow these steps to use the search engine:

### Step 1: Enter a Search Query
When you see `Enter search query (or 'exit' to quit):`, type keywords related to the news you want to find.
* **Example:** `Sania Mirza`
* **Example:** `World Bank Economy`
* **Example:** `FIFA`

### Step 2: View Results
The system will display the **Top 3** most relevant articles, including:
* **[Rank]:** The position of the result (1 is the best match).
* **Title:** The headline of the news article.
* **Score:** The BM25 relevance score (a higher number means a better match).
* **Preview:** A short snippet of the article text to help you identify the content.

### Step 3: Exit the Program
To stop the search engine, simply type one of the following commands and press Enter:
* `exit`
* `quit`
````
