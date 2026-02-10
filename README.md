# 👁️ Hybrid AI Image Analyzer
 
> A professional Python application leveraging **Azure Computer Vision (v3.2)** for technical detection and **Azure OpenAI (GPT-4)** for human-like image summarization.

---

## 🚀 Project Overview

This application bridges the gap between traditional Computer Vision and Generative AI. It processes raw image data to extract technical metadata (objects, tags) and feeds that data into a Large Language Model to generate accessible, human-readable summaries.

### 🔑 Key Features (Assignment Requirements)

This project strictly adheres to the provided engineering requirements:

1.  **Binary Data Handling** ✅
    *   Images are processed in memory as binary streams (`bytes`). No temporary files are saved to disk, ensuring security and speed.
2.  **JSON Parsing** ✅
    *   The `analyzer.py` module parses complex JSON responses from Azure, extracting deep nested fields like `description.captions[0].text` and `objects[].confidence`.
3.  **Probabilistic Decisions** ✅
    *   The system implements decision logic:
        *   **> 80%**: High Confidence (Green)
        *   **50-80%**: Moderate Confidence (Yellow)
        *   **< 50%**: Uncertain (Red)
4.  **Uncertainty Handling** ✅
    *   The UI actively warns users if the AI is unsure. Low-confidence results trigger specific alert boxes and visual cues in the interface.

---

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) (Dark Mode UI)
*   **Backend Logic:** Python 3.8+
*   **AI Service 1:** Azure Computer Vision (v3.2) - *Feature Extraction*
*   **AI Service 2:** Azure OpenAI (GPT-4o) - *Natural Language Synthesis*
*   **Security:** `python-dotenv` for environment variable management.

---

## 📂 Project Structure

The project is modularized into **UI** and **Business Logic**:

```text
hybrid-analyzer/
│
├── main.py              # Frontend: Streamlit UI & State Management
├── analyzer.py         # Backend: Class-based API handling & Logic
│
├── .env                # API Keys (Excluded from Version Control)
├── requirements.txt    # Project Dependencies
└── README.md           # Documentation
```
## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Image_Analyzer
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root directory. Add your Azure credentials:

```ini
# --- AZURE COMPUTER VISION ---
AZURE_VISION_ENDPOINT="https://<your-resource-name>.cognitiveservices.azure.com/"
AZURE_VISION_KEY="<your-vision-key>"

# --- AZURE OPENAI ---
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_KEY="<your-openai-key>"
AZURE_OPENAI_DEPLOYMENT="<your-deployment-name>" 
AZURE_OPENAI_VERSION="2024-02-15-preview"
```

### 5. Run the Application
```bash
streamlit run src/main.py
```
