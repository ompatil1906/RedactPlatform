🛡️ RedactPlatform

RedactPlatform is a powerful Streamlit-based web application built for privacy-focused text and PDF data redaction. Leveraging Microsoft Presidio, OCR (Tesseract, NanoNets, Whisper), and OpenAI LLMs, the platform identifies and redacts Personally Identifiable Information (PII) from documents in multiple formats.

🚀 Features

🔍 Entity Recognition via Presidio Analyzer
📄 PDF redaction support (PyPDF2 + PyMuPDF)
📸 OCR integration with:
Tesseract
NanoNets
Whisper
🤖 LLM-powered fake data generation for redacted entities (OpenAI)
🧠 Azure AI Text Analytics fallback support
🧱 Annotated view using st-annotated-text
💡 Built with Streamlit for fast UI
🧰 Tech Stack

Category	Libraries/Tools
NLP & Redaction	presidio-analyzer, presidio-anonymizer
UI	streamlit, streamlit-tags, st-annotated-text
OCR	pytesseract, NanoNets, Whisper
LLMs	openai, langchain, azure-ai-textanalytics
PDF Processing	PyPDF2, PyMuPDF
Backend	Python 3.10 + Poetry
⚙️ Installation

🔗 Requirements
Python 3.10+
Poetry (or use pip with requirements.txt)
Tesseract installed and configured in PATH (for OCR)
🔧 Setup
Option 1: Poetry

git clone https://github.com/ompatil1906/RedactPlatform.git
cd RedactPlatform

poetry install
poetry shell
streamlit run presidio_streamlit.py
Option 2: Pip

pip install -r requirements.txt
streamlit run presidio_streamlit.py
📁 Project Structure

├── presidio_streamlit.py       # Main Streamlit app
├── presidio_helpers.py         # Presidio logic helpers
├── presidio_nlp_engine_config.py
├── openai_fake_data_generator.py
├── nanonets_ocr.py
├── llm_whisper.py              # Whisper-based OCR
├── requirements.txt
├── pyproject.toml / poetry.lock
├── .env                        # For API keys (OpenAI, NanoNets, etc.)
🧪 Usage

Run the app:
streamlit run presidio_streamlit.py
Upload a text or PDF file.
Choose redaction method and models (Presidio / LLM / OCR).
View and download the redacted output.
🔐 .env Configuration

Create a .env file with your API keys:

OPENAI_API_KEY=your-openai-key
NANONETS_API_KEY=your-nanonets-key
AZURE_API_KEY=your-azure-key
AZURE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
📜 License

MIT License. See LICENSE file.

