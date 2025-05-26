# RAG Document Expansion MRR

A comprehensive evaluation system for comparing retrieval performance between original documents and expanded documents in Retrieval-Augmented Generation (RAG) systems using Mean Reciprocal Rank (MRR) metrics.

## Project Overview

This project implements and evaluates document expansion techniques for improving retrieval accuracy in RAG systems. It compares three different approaches:

1. **Original text only** - Documents in their original form
2. **Original + Expanded text** - Combination of original and AI-generated expanded content
3. **Expanded text only** - Only the AI-generated expanded content

The system uses Qdrant as the vector database and Google Gemini for content generation and expansion.

## Project Structure

```
rag-document-expansion-mrr/
├── main.py                     # Main execution interface
├── main_example.py             # Testing and examples
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Qdrant setup
├── LICENSE                     # MIT License
├── README.md                   # This file
├── data/                       # Data storage
│   ├── data_set/              # Original datasets (benefits, performance_review, etc.)
│   ├── expanded/              # AI-expanded documents
│   ├── formatted/             # Formatted text documents
│   ├── mrr_results/           # MRR evaluation results
│   ├── question/              # Generated test questions
│   └── report/                # Generated reports
└── src/                       # Source code
    ├── constants/             # Configuration constants
    ├── embeddings/            # Text embedding utilities
    ├── evaluation/            # MRR and similarity evaluation
    ├── executions/            # Main execution modules
    ├── generation/            # LLM content generation
    ├── models/                # Data models and schemas
    ├── retrieval/             # Vector database operations
    └── utils/                 # Utility functions
```

## Key Features

- **Document Expansion**: AI-powered document expansion using Google Gemini
- **Vector Storage**: Efficient document storage and retrieval using Qdrant
- **MRR Evaluation**: Comprehensive Mean Reciprocal Rank evaluation system
- **Question Generation**: Automated test question generation for evaluation
- **Comparative Analysis**: Performance comparison across different text representation approaches
- **Report Generation**: Automated markdown report generation

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Qdrant)
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-document-expansion-mrr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Qdrant using Docker:
```bash
docker-compose up -d
```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your API key:
```bash
GOOGLE_GEMINI_API_KEY=your-api-key-here
```

## Usage

### Main Interface ([main.py](main.py))

The main interface provides a menu-driven system for executing different components:

```bash
python main.py
```

**Available Options:**
- **0**: Check your options
- **1**: Insert original text into Qdrant
- **2**: Insert original + expanded text into Qdrant  
- **3**: Insert only expanded text into Qdrant
- **4**: Generate questions
- **5**: Evaluate similarity search
- **6**: Evaluate MRR
- **7**: Generate MRR comparison report
- **99**: Exit

**Example workflow:**
1. Start with option **1** to insert original documents
2. Use option **2** to insert combined original + expanded documents
3. Use option **3** to insert only expanded documents
4. Generate test questions with option **4**
5. Evaluate performance with options **5** and **6**
6. Generate comparison report with option **7**

### Testing Interface ([main_example.py](main_example.py))

The example file contains various testing functions for development and debugging:

```bash
python main_example.py
```

**Available Tests:**
- **Dense/Sparse Embeddings**: Test text embedding functionality
- **Qdrant Operations**: Test vector database connections and operations
- **Document Insertion**: Test adding documents to collections
- **Similarity Search**: Test retrieval functionality
- **LLM Integration**: Test Google Gemini content generation

**Key Test Functions:**
- [`test_dense_embedding()`](main_example.py) - Test dense vector embeddings
- [`test_qdrant_connection()`](main_example.py) - Verify Qdrant connectivity
- [`test_similarity_search()`](main_example.py) - Test document retrieval
- [`test_llm_with_model()`](main_example.py) - Test structured LLM responses

## Core Components

### Document Processing
- **Dataset Generation**: Creates synthetic HR documents using [`src/executions/generate_data_set.py`](src/executions/generate_data_set.py)
- **Document Expansion**: Generates expanded content using [`src/executions/insert_documents.py`](src/executions/insert_documents.py)
- **Question Generation**: Creates test queries using [`src/executions/generate_question.py`](src/executions/generate_question.py)

### Evaluation System
- **MRR Calculation**: Implements Mean Reciprocal Rank evaluation in [`src/executions/evaluate_mrr.py`](src/executions/evaluate_mrr.py)
- **Similarity Evaluation**: Performance testing via [`src/executions/evaluate_similarity_search.py`](src/executions/evaluate_similarity_search.py)
- **Report Generation**: Automated reporting through [`src/executions/generate_report.py`](src/executions/generate_report.py)

### Data Models
- **Document Models**: Defined in [`src/models/generated_document.py`](src/models/generated_document.py)
- **MRR Models**: Evaluation schemas in [`src/models/mrr_document.py`](src/models/mrr_document.py)
- **Question Models**: Query structures for testing

## Results

The system generates comprehensive performance comparisons. Based on the latest evaluation results in [`data/report/mrr_comparison_report.md`](data/report/mrr_comparison_report.md):

| Dataset                    | Avg MRR@1 | Avg MRR@3 | Avg MRR@5 | Avg MRR@10 |
|:---------------------------|----------:|----------:|----------:|-----------:|
| original_text              |    0.94   |   0.9533  |   0.9573  |    0.9573  |
| original_and_expanded_text |    0.96   |   0.9733  |   0.9733  |    0.9733  |
| only_expanded_text         |    0.90   |   0.9333  |   0.9333  |    0.9333  |

**Key Finding**: The **original_and_expanded_text** approach achieves the best performance with MRR@1 of 0.9600.

## Configuration

Key configuration files:
- [`src/constants/constants.py`](src/constants/constants.py) - API keys, model names, and system constants
- [`.env`](.env) - Environment variables (Google Gemini API key)
- [`docker-compose.yml`](docker-compose.yml) - Qdrant vector database setup
- [`requirements.txt`](requirements.txt) - Python package dependencies

## Data Flow

1. **Data Generation**: Create synthetic documents using Google Gemini
2. **Document Expansion**: Generate question-based expansions for each document
3. **Vector Storage**: Store embeddings in Qdrant collections
4. **Question Generation**: Create diverse test queries
5. **Evaluation**: Run MRR and similarity evaluations
6. **Reporting**: Generate comparative performance reports

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.