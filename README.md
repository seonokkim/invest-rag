# INVEST-RAG: Investment Education Question Answering System

INVEST-RAG is a Retrieval-Augmented Generation (RAG) system that provides educational information about investments and stock markets. The system combines modern NLP techniques with investment knowledge to provide beginner-friendly, contextual responses about finance and investing.

## Features

- **RAG Architecture**: Retrieves relevant investment education passages based on semantic similarity to user queries
- **LLM Integration**: Uses meta-llama/Llama-3.2-3B-Instruct model for generating responses
- **Efficient Retrieval**: Employs sentence transformers for creating and comparing embeddings
- **User-Friendly Interface**: Built with Streamlit for an accessible web application experience
- **Performance Optimized**: Uses 4-bit quantization for efficient model deployment

## Project Structure

```
invest-rag/
├── invest_app.py       # Main Streamlit application
├── clean_text.pkl      # Preprocessed investment education text segments
├── invest_embeddings.pt # Pre-computed embeddings for investment texts
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/seonokkim/invest-rag.git
   cd invest-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary data files:
   - `clean_text.pkl`: Preprocessed investment education text segments
   - `invest_embeddings.pt`: Pre-computed embeddings for investment text segments

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run invest_app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter your question about investments or stock markets in the text area

4. Click "Get Answer" to receive a response based on relevant investment education passages

5. Expand the "See Retrieved Context" section to view the source passages used for the response

## Technical Details

### Models Used

- **LLM**: meta-llama/Llama-3.2-3B-Instruct
- **Embedding Model**: SentenceTransformer's all-MiniLM-L12-v2

### RAG Flow

1. The user query is converted to an embedding vector
2. This embedding is compared to pre-computed embeddings of investment education text segments
3. The most semantically similar segments are retrieved 
4. These segments are provided as context to the LLM
5. The LLM generates a coherent, contextually informed response that is:
   - Beginner-friendly (understandable by high school students)
   - Based only on the provided investment knowledge
   - Limited to approximately 200 words

### Optimization

- 4-bit quantization for reduced memory footprint
- Resource caching for improved performance on repeated queries
- Selective context retrieval based on similarity threshold

## Contributing

Contributions to improve INVEST-RAG are welcome! Please feel free to submit a Pull Request on the [GitHub repository](https://github.com/seonokkim/invest-rag).

## License

MIT License

## Acknowledgements

- Educational investment content used in this project
- [NLP Repository](https://github.com/mohan696matlab/NLP) by mohan696matlab for inspiration and techniques
- HuggingFace for providing access to transformer models
- The open-source libraries that made this project possible

## Contact

For questions or feedback, please contact:
- GitHub: [seonokkim](https://github.com/seonokkim)
- Email: seonokrkim@gmail.com
