# 🌱 AgriMind AI RAG Chatbot

**AgriMind** is an AI-powered agricultural disease advisory assistant that combines **crop disease image classification**, **Retrieval-Augmented Generation (RAG)**, **FAISS semantic search**, and the **Google Gemini API** to provide evidence-based agricultural information for extension workers.

The system currently supports **maize and wheat disease identification** and uses a curated agricultural knowledge base to retrieve relevant evidence before generating an answer.

## 🚀 Live Demo

**Streamlit App:**
https://agrimindairagchatbot-8euydufbhnu7bpzdvdakeo.streamlit.app/

> **Note:** The application uses the Gemini API for natural-language answer generation. If the Gemini API quota is temporarily exhausted, the AI-generated response may be unavailable even though agricultural evidence retrieval continues to work.

---

## 🎯 Project Objective

Agricultural extension workers often need quick access to reliable information about crop diseases, their symptoms, transmission, prevention, and management.

AgriMind aims to provide a practical AI assistant that:

* Identifies crop diseases from leaf images.
* Retrieves relevant agricultural evidence from a curated knowledge base.
* Uses an LLM to generate concise answers grounded in the retrieved evidence.
* Distinguishes Ethiopian-specific evidence from general international references.
* Avoids unsupported pesticide recommendations.
* Clearly communicates uncertainty when evidence is insufficient.

---

## 🧠 System Architecture

```text
                 ┌─────────────────────┐
                 │     Leaf Image      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Disease Classifier  │
                 │  Maize / Wheat      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Disease Prediction  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   RAG Query Builder │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Sentence Transformer│
                 │  all-MiniLM-L6-v2   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    FAISS Search     │
                 │  Semantic Retrieval │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Agricultural        │
                 │ Knowledge Base      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Retrieved Evidence│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Google Gemini    │
                 │   Answer Generation │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Extension Worker    │
                 │      Response       │
                 └─────────────────────┘
```

---

## ✨ Key Features

### 🌾 Crop Disease Classification

The application accepts crop leaf images and uses trained deep-learning models to identify diseases.

Currently supported crops include:

* **Maize**
* **Wheat**

The trained models are stored in the `models/` directory.

### 🔎 Retrieval-Augmented Generation

AgriMind does not rely solely on the LLM's internal knowledge.

Instead, the system:

1. Receives the user's question.
2. Builds a disease-specific RAG query.
3. Converts the query into an embedding.
4. Searches the agricultural knowledge base using FAISS.
5. Retrieves the most relevant evidence.
6. Passes the retrieved evidence to Gemini.
7. Generates an answer grounded in the retrieved evidence.

This helps reduce unsupported or fabricated information.

### 📚 Evidence Display

The application displays the retrieved evidence together with:

* Retrieval score
* Crop
* Disease
* Topic
* Source
* Organization/author
* Publication year
* Source type
* Region
* Evidence type
* Confidence

This makes the information used by the RAG system transparent to the user.

### 🇪🇹 Ethiopia-Aware Guidance

The system is designed to distinguish:

* Ethiopian agricultural evidence
* Regional African evidence
* General international technical references

International recommendations are not automatically treated as Ethiopian decision rules.

### ⚠️ Safety-Aware Agricultural Advice

AgriMind is designed to avoid unsupported chemical recommendations.

The system does not provide:

* Unsupported pesticide products
* Unsupported application rates
* Unsupported doses
* Unsupported spray schedules

unless such information is explicitly supported by appropriate current Ethiopian guidance in the retrieved evidence.

The system also emphasizes that an image-classifier prediction should **not automatically be considered a confirmed laboratory diagnosis**.

---

## 🛠️ Technologies Used

| Component            | Technology                |
| -------------------- | ------------------------- |
| User Interface       | Streamlit                 |
| Programming Language | Python                    |
| Deep Learning        | TensorFlow / Keras        |
| RAG Framework        | Custom RAG pipeline       |
| Embeddings           | Sentence Transformers     |
| Embedding Model      | `all-MiniLM-L6-v2`        |
| Vector Search        | FAISS                     |
| LLM                  | Google Gemini API         |
| Knowledge Base       | JSONL                     |
| Source Control       | Git / GitHub              |
| Deployment           | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
agrimind_ai_rag_chatbot/
│
├── app.py
│
├── requirements.txt
├── pyproject.toml
├── runtime.txt
├── .gitignore
│
├── knowledge_base.jsonl
│
├── models/
│   ├── maize/
│   └── wheat/
│
└── src/
    ├── __init__.py
    ├── classifier.py
    ├── retrieval.py
    ├── generator.py
    └── ...
```

---

## 📖 Knowledge Base

The knowledge base is stored in:

```text
knowledge_base.jsonl
```

Each record contains structured information such as:

```json
{
  "crop": "maize",
  "disease": "MLN",
  "topic": "description",
  "content": "...",
  "source_id": "S1",
  "source_title": "...",
  "source_organization": "...",
  "publication_year": "2020",
  "source_type": "Peer-reviewed review",
  "source_url": "...",
  "region": "Sub-Saharan Africa; Ethiopia included",
  "evidence_type": "authoritative/peer-reviewed",
  "confidence": "high"
}
```

This structure allows the retrieval system to preserve the provenance and context of agricultural evidence.

---

## 🔬 RAG Retrieval

The retrieval system uses:

```text
SentenceTransformer
        ↓
all-MiniLM-L6-v2
        ↓
Normalized embeddings
        ↓
FAISS IndexFlatIP
        ↓
Similarity search
```

Retrieved documents are ranked according to their semantic similarity to the user's question.

For example:

```text
User Question
     ↓
"How can I manage Septoria in wheat?"
     ↓
Embedding
     ↓
FAISS
     ↓
Top relevant evidence
     ↓
Gemini
     ↓
Evidence-grounded response
```

---

## 🤖 Gemini Integration

The application uses the Google Gemini API to generate the final natural-language response.

The API key is loaded through an environment variable:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

The API key should **never be committed to GitHub**.

For Streamlit Cloud, the key should be stored using the application's **Secrets** configuration.

Example:

```text
GEMINI_API_KEY = "your-api-key"
```

---

## ⚙️ Local Installation

Clone the repository:

```bash
git clone https://github.com/fevenabebe/agrimind_ai_rag_chatbot.git
```

Move into the project:

```bash
cd agrimind_ai_rag_chatbot
```

Create a Python environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment variable:

```bash
export GEMINI_API_KEY="your-api-key"
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔐 Security

API keys and credentials must not be committed to the repository.

Use environment variables or Streamlit Secrets instead.

The `.gitignore` file should exclude sensitive files such as:

```text
.env
.env.*
__pycache__/
venv/
.venv/
```

---

## ⚠️ Limitations

AgriMind is a research and demonstration system and has several limitations.

### API Dependency

The answer-generation component depends on the Gemini API.

If the API quota is exhausted or the service is unavailable, the LLM-generated response cannot be produced.

### Image Classification

The disease classifier provides a model prediction rather than a definitive laboratory diagnosis.

### Knowledge Base Coverage

The quality of the RAG response depends on the coverage and quality of the agricultural evidence contained in the knowledge base.

### Ethiopian Recommendations

Not all agricultural diseases currently have sufficiently detailed Ethiopian official guidance in the knowledge base.

Therefore, the system may recommend consultation with:

* Ethiopian Institute of Agricultural Research (EIAR)
* Ministry of Agriculture
* Regional agricultural bureaus
* Qualified agricultural extension workers
* Plant pathology specialists

when the available evidence is insufficient.

---

## 🧪 Example Use Case

An extension worker uploads a wheat leaf image.

The classifier predicts:

```text
Crop: Wheat
Disease: Septoria
```

The worker then asks:

```text
How can I manage this disease?
```

AgriMind retrieves relevant evidence about:

* Disease management
* Disease-favorable conditions
* Symptoms
* Differential diagnosis
* Resistance
* Relevant warnings

The retrieved evidence is then passed to Gemini to generate a concise response.

---

## 🌱 Current Supported Diseases

The knowledge base currently includes agricultural evidence for diseases including:

### Maize

* Maize Lethal Necrosis (MLN)
* Maize Streak Virus (MSV)

### Wheat

* Wheat Septoria

The knowledge base can be expanded with additional crops, diseases, and authoritative agricultural sources.

---

## 🔮 Future Improvements

Planned improvements include:

* [ ] Expand the agricultural knowledge base.
* [ ] Add more Ethiopian-specific sources.
* [ ] Add multilingual support.
* [ ] Improve retrieval ranking.
* [ ] Add stronger evidence filtering.
* [ ] Add confidence-aware responses.
* [ ] Add a robust LLM fallback mechanism.
* [ ] Add additional crop disease models.
* [ ] Improve model evaluation and monitoring.
* [ ] Add more comprehensive source citations.
* [ ] Improve production reliability and API management.

---

## 👩‍💻 Project

**AgriMind AI RAG Chatbot**

An AI engineering project combining:

**Computer Vision + RAG + Vector Search + LLMs + Agricultural Knowledge**

### Live Application

https://agrimindairagchatbot-8euydufbhnu7bpzdvdakeo.streamlit.app/

---

## ⚖️ Disclaimer

AgriMind is intended to support agricultural information access and extension work. It should not replace professional plant pathology diagnosis, official agricultural recommendations, or laboratory confirmation.

Disease predictions from the image-classification component should be treated as **model predictions**, not definitive diagnoses.

For chemical control and other regulated agricultural interventions, users should follow current recommendations from appropriate Ethiopian authorities and qualified agricultural professionals.
