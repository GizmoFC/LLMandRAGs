# AI-Powered Collaborative Contract Editor

A comprehensive system for intelligent contract analysis and clause retrieval using RAG (Retrieval-Augmented Generation) with Ollama and FAISS vector search.

## Project Structure

```
datastuff/
│
├── data/
│   ├── synthetic_clause_library.jsonl   # Hand-curated contract clauses
│   └── cuad_clause_library.jsonl        # CUAD dataset converted to JSONL
│
├── embeddings/
│   └── embed_clauses.py                 # Loads JSONL → Embeds → FAISS
│
├── ollama_server/
│   └── ollama_qa.py                     # RAG pipeline with LangChain + Ollama
│
├── utils/
│   └── convert_cuad_to_jsonl.py         # Converts CUAD HuggingFace → JSONL
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Features

- **Intelligent Clause Retrieval**: Find relevant contract clauses using semantic search
- **RAG Pipeline**: Question-answering system powered by Ollama and LangChain
- **Vector Search**: FAISS-based similarity search for fast clause matching
- **Multi-Source Data**: Support for both synthetic and public contract datasets
- **Easy Setup**: Simple installation and configuration process

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Git

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd datastuff
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama**:
   ```bash
   # Download from https://ollama.ai/
   # Or use the installer for your platform
   
   # Pull a model (e.g., llama2)
   ollama pull llama2
   
   # Start Ollama server
   ollama serve
   ```

## Setup Instructions

### 1. Prepare Your Data

**Option A: Use synthetic data**
- The `synthetic_clause_library.jsonl` file contains hand-curated contract clauses
- Each line is a JSON object with `text` and `metadata` fields

**Option B: Convert CUAD dataset**
```bash
python utils/convert_cuad_to_jsonl.py
```

### 2. Generate Embeddings

```bash
python embeddings/embed_clauses.py
```

This will:
- Load clauses from JSONL files
- Generate embeddings using sentence-transformers
- Store vectors in FAISS index
- Save the index to disk

### 3. Start the RAG Server

```bash
python ollama_server/ollama_qa.py
```

The server will:
- Load the FAISS index
- Start a web interface for Q&A
- Connect to Ollama for text generation

## Usage

### Web Interface
- Navigate to `http://localhost:8000` (or the port specified)
- Enter your contract-related questions
- Get intelligent answers with relevant clause citations

### API Endpoints
- `POST /query` - Submit a question
- `GET /health` - Check server status
- `GET /clauses` - List available clauses

### Example Queries
- "What are the termination clauses in this contract?"
- "Find indemnification provisions"
- "What are the payment terms?"
- "Show me force majeure clauses"

## Configuration

### Environment Variables
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `MODEL_NAME`: Ollama model to use (default: llama2)
- `FAISS_INDEX_PATH`: Path to FAISS index file
- `EMBEDDING_MODEL`: Sentence transformer model name

### Model Selection
The system works with any Ollama model. Recommended models:
- `llama2` - Good balance of speed and quality
- `llama2:13b` - Higher quality, slower inference
- `mistral` - Fast and efficient
- `codellama` - Good for technical contracts

## Data Format

### JSONL Structure
Each line in the JSONL files should contain:
```json
{
  "text": "The actual clause text...",
  "metadata": {
    "clause_type": "termination",
    "contract_type": "employment",
    "jurisdiction": "US",
    "source": "synthetic"
  }
}
```

### Supported Clause Types
- termination
- indemnification
- force_majeure
- payment_terms
- confidentiality
- intellectual_property
- dispute_resolution
- liability_limitations

## Troubleshooting

### Common Issues

1. **Ollama connection failed**
   - Ensure Ollama is running: `ollama serve`
   - Check the model is downloaded: `ollama list`

2. **FAISS index not found**
   - Run the embedding generation script first
   - Check file paths in configuration

3. **Memory issues with large datasets**
   - Use smaller embedding models
   - Process data in batches
   - Consider using GPU acceleration

### Performance Optimization

- Use GPU for embedding generation if available
- Adjust FAISS index parameters for your dataset size
- Consider using quantized models for faster inference
- Implement caching for frequent queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CUAD dataset for public contract data
- Ollama team for the local LLM server
- LangChain for the RAG framework
- FAISS for efficient vector search 