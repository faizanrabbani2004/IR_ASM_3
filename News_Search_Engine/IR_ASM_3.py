import os
import csv
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
DATA_FILE = "data.csv"  # The name of your CSV file

# --- 1. SETUP & PREPROCESSING ---
def setup_nltk():
    """Ensures necessary NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def process(self, text):
        """
        Pipeline: Lowercase -> Remove Punctuation -> Tokenize -> Remove Stopwords -> Stemming
        """
        if not text:
            return []
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove Punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # 3. Tokenize
        tokens = word_tokenize(text)
        
        # 4. Remove Stopwords & Stemming
        clean_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return clean_tokens

# --- 2. THE RETRIEVAL ENGINE ---
class LocalIRSystem:
    def __init__(self, data_file):
        self.data_file = data_file
        self.documents = []   # Stores raw text for display
        self.doc_titles = []  # Stores headings/titles
        self.corpus = []      # Stores processed tokens for the algorithm
        self.preprocessor = TextPreprocessor()
        self.bm25 = None      # The Index
        
    def ingest_data(self):
        """Reads the CSV file."""
        if not os.path.exists(self.data_file):
            print(f"Error: File '{self.data_file}' not found.")
            return

        print(f"Loading data from {self.data_file}...")
        
        with open(self.data_file, mode='r', encoding='utf-8', errors='ignore') as csvfile:
            # Using DictReader to handle headers automatically
            reader = csv.DictReader(csvfile)
            
            count = 0
            for row in reader:
                # We use the 'Article' column for the content and 'Heading' for the title
                content = row.get('Article', '')
                heading = row.get('Heading', 'Untitled')
                
                # Skip empty articles
                if not content.strip():
                    continue

                self.doc_titles.append(heading)
                self.documents.append(content)
                
                # Preprocess immediately
                self.corpus.append(self.preprocessor.process(content))
                count += 1
        
        print(f"Data ingestion complete. Loaded {count} articles.")

    def build_index(self):
        """Builds the BM25 Index."""
        if not self.corpus:
            print("No data to index.")
            return
        print("Building BM25 Index...")
        self.bm25 = BM25Okapi(self.corpus)
        print("Indexing complete.")

    def search(self, query, top_k=3):
        """Retrieves and ranks documents based on the query."""
        if not self.bm25:
            return []
        
        tokenized_query = self.preprocessor.process(query)
        
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Zip scores with titles and contents
        results = zip(self.doc_titles, self.documents, doc_scores)
        
        # Sort by score (descending)
        ranked_results = sorted(results, key=lambda x: x[2], reverse=True)
        
        return ranked_results[:top_k]

# --- 3. MAIN EXECUTION LOOP ---
def main():
    setup_nltk()
    
    # Initialize System
    system = LocalIRSystem(DATA_FILE)
    
    # 1. Ingest
    start_time = time.time()
    system.ingest_data()
    
    # 2. Index
    system.build_index()
    
    if not system.documents:
        print("System failed to load documents. Please check your CSV file.")
        return

    print(f"System ready in {time.time() - start_time:.4f} seconds.\n")
    
    # 3. Search Loop
    while True:
        print("-" * 50)
        query = input("Enter search query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
            
        start_search = time.time()
        results = system.search(query)
        duration = time.time() - start_search
        
        print(f"\nFound {len(results)} results in {duration:.4f} seconds:")
        for rank, (title, content, score) in enumerate(results, 1):
            if score > 0:
                # Show first 200 chars as preview
                preview = content[:200].replace('\n', ' ') + "..."
                print(f"[{rank}] {title}")
                print(f"    Score: {score:.4f}")
                print(f"    Preview: {preview}\n")
            else:
                # If score is 0, it means no words matched
                pass

if __name__ == "__main__":
    main()