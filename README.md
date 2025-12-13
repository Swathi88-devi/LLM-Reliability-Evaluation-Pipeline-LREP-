**LLM Evaluation Pipeline**

**Overview**

This repository features a streamlined LLM response evaluation pipeline that automatically reviews and scores model outputs at scale. The system checks for relevance, completeness, and any potential hallucinations by comparing model responses to provided context documents. It's designed to be quick and cost-effective, making it perfect for real-time and large-scale use—think millions of conversations daily.

**Local Setup Instructions**

Prerequisites
Python 3.10+
pip

**Steps**

**Clone the repository:**

git clone https://github.com/Swathi88-devi/LLM-Reliability-Evaluation-Pipeline-LREP-.git

cd llm-evaluation-pipeline

**(Optional) Set up a virtual environment:**

python -m venv .venv

source .venv/bin/activate  # For Windows: .venv\Scripts\activate

**Install the required dependencies:**

pip install -r requirements.txt

**Put your input files in the input/ folder:**

conversation.csv (model responses)
Optional context files (CSV/JSON)

**Run the pipeline:**

python code/run.py

**You’ll find the outputs in the output/ folder:**

batch_results.json
evaluation_summary.csv
**How the Pipeline Works**

The system supports both CSV and JSON inputs for conversations and context documents:

1. Conversation Input (CSV or JSON)
This contains the user’s question, the model’s response, and optional metadata such as timestamps or token counts.

2. Context Input (CSV or JSON)
This includes a list of context documents (typically retrieved from a vector database), where each entry contains the text used for grounding the evaluation.

These documents help verify whether the model stayed grounded or went off the rails.


**Project Highlights**

- End-to-end LLM evaluation pipeline
- Covers semantic scoring, hallucination detection, and keyword completeness
- Measures latency + token-based cost
- Modular architecture (easy to extend)
- Clean Input/Output folder structure for quick testing

**Architecture of the Evaluation Pipeline**

Input Data (conversation.csv)         
       |         
       v  
Preprocessing Module  - Text cleaning  - Sentence splitting 
        |         
        v  
 Evaluation Engine 
- Sentence-level embeddings (SBERT)
- Cosine similarity with context
- Keyword completeness check
-  Hallucination detection
         |
         v
   Aggregation & Scoring
   - Relevance score
   - Completeness score
   - Flags for unreliable sentences
            |
            v
     Output Generation
     - batch_results.json
     - evaluation_summary.csv
The pipeline is modular and runs entirely on embeddings, which makes it easy to integrate with Retrieval-Augmented Generation (RAG) systems or conversational AI platforms.

**Core Evaluation Logic**

**1. Relevance Score**

The response is broken down into individual sentences, each embedded using an SBERT model. Every sentence is then compared with all context vectors using cosine similarity.A sentence that closely matches any part of the context will have a high similarity score, which directly influences the relevance evaluation.

**2. Completeness Score**

The pipeline pulls out basic keywords from the context documents and checks how many of those show up in the LLM’s response. It’s a straightforward way to assess whether the answer addressed the key points from the retrieved information.

**3. Hallucination Check**

If a sentence’s similarity score dips below a certain threshold (0.35 by default), it’s flagged as a potential hallucination. For each flagged sentence, the pipeline logs:
the sentence text
its highest similarity score
This gives a clearer picture of which parts of the response might be unreliable.

**4. Latency (Optional)**

If start and end timestamps are available, the pipeline calculates the time it took for the model to respond. This can be helpful when comparing different LLMs under similar loads.

**5. Token and Cost Estimation**

If token counts aren’t provided, the system estimates them using the tiktoken library or a backup method. The cost is calculated with a simple formula:
cost_usd = (tokens / 1000) * cost_per_1k_tokens.

**Design Decisions (Why This Approach)**

This approach is intentionally designed to be simple, fast, and budget-friendly:

- Sentence-level evaluation helps prevent irrelevant or hallucinated sentences from skewing overall scores.

- Embedding-based similarity avoids costly secondary LLM requests while ensuring semantic accuracy.

- Lightweight SBERT models (like MiniLM) strike a great balance between speed and performance.

- Minimal preprocessing cuts down on overhead and makes debugging simpler.

- Plain CSV/JSON I/O allows easy integration into current data pipelines.
We steered clear of alternatives that relied on extra LLM-based verification because they typically involve higher latency and costs, which wouldn't work well for real-time or high-volume scenarios.

**Scalability, Latency & Cost Optimization**

- The pipeline is set up to handle millions of evaluations each day with minimal operational costs:

- Batch embedding computation boosts throughput and lessens per-request overhead.

- Small embedding models greatly lower inference latency.

- No extra LLM calls are needed during evaluation, keeping expenses in check.

- Pure embedding-based logic makes scaling across workers easy.

- Simple data formats support parallel processing and quick serialization/deserialization.

- These choices keep the system responsive and cost-efficient, even under heavy workloads.

**Project Structure**

llm-evaluation-pipeline/

├── code/

│ ├── pipeline.py

│ └── run.py

├── input/

│ └── conversation.csv

├── output/

│ ├── batch_results.json

│ └── evaluation_summary.csv

├── requirements.txt

└── README.md

**Example Output**

{  "relevance_score": 0.81,  "completeness_score": 0.62,  "hallucination_detected": true,  "hallucinated_sentences": [    {      "sentence": "The product was launched in 1995.",      "max_similarity": 0.21    }  ],  "latency_ms": 128,  "estimated_tokens": 42,  "estimated_cost_usd": 0.00008}

**Technologies Used**

Python 3.10+

Sentence Transformers (SBERT)

NumPy, Pandas

scikit-learn

tiktoken

Basic logging

JSON / CSV

**Notes**

The code follows PEP-8 guidelines where relevant.

This repository is open for public review.

The solution is made to be easily extendable for more metrics or evaluation strategies.

**Conclusion**

This project combines a variety of lightweight techniques to evaluate LLM responses in a structured, understandable way. It checks quality, factual accuracy, completeness, and cost-efficiency of model outputs without depending on heavy LLM calls. The pipeline is flexible enough to integrate into a production RAG system and straightforward enough to run locally during development.

**Future Improvements**

- Support for comparing multiple models

- Advanced keyword extraction (like RAKE/YAKE)

- A dashboard for visualizing evaluation metrics

- Streaming capabilities for real-time evaluations



**Author: Perabattula Swathi Naga Devi**
