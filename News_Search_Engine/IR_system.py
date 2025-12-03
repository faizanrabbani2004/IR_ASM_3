import os
import csv
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

DATA_FILE = "data.csv"

def setup_nltk():

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
        if not text:
            return []
            
        
        text = text.lower()
        

        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        tokens = word_tokenize(text)
        
    
        clean_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return clean_tokens


class LocalIRSystem:
    def __init__(self, data_file):
        self.data_file = data_file
        self.documents = []
        self.doc_titles = []
        self.corpus = []
        self.preprocessor = TextPreprocessor()
        self.bm25 = None
        
    def ingest_data(self):
        if not os.path.exists(self.data_file):
            print(f"Error: File '{self.data_file}' not found.")
            return

        print(f"Loading data from {self.data_file}...")
        
        with open(self.data_file, mode='r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                content = row.get('Article', '')
                heading = row.get('Heading', 'Untitled')
                if not content.strip():
                    continue

                self.doc_titles.append(heading)
                
                self.documents.append(content)
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
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        results = zip(self.doc_titles, self.documents, doc_scores)
        

        ranked_results = sorted(results, key=lambda x: x[2], reverse=True)
        
        return ranked_results[:top_k]


def main():
    setup_nltk()
    
 
    system = LocalIRSystem(DATA_FILE)
    
 
    start_time = time.time()
    system.ingest_data()
    
  
    system.build_index()
    
    if not system.documents:
        print("System failed to load documents. Please check your CSV file.")
        return

    print(f"System ready in {time.time() - start_time:.4f} seconds.\n")
    

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
                preview = content[:200].replace('\n', ' ') + "..."
                print(f"[{rank}] {title}")
                print(f"    Score: {score:.4f}")
                print(f"    Preview: {preview}\n")
            else:
                pass

if __name__ == "__main__":
    main()