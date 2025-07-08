🛡️ RedactPlatform
RedactPlatform is a powerful Streamlit-based web application for PII detection and redaction from text and PDF documents. It leverages Microsoft Presidio, OCR engines (Tesseract, NanoNets, Whisper), and LLMs (OpenAI) to identify and anonymize sensitive information in documents.

🚀 Features

🔍 Entity Detection: Powered by Presidio Analyzer for robust PII identification.
📄 PDF Redaction: Supports redaction in PDF files using PyPDF2 and PyMuPDF.
📸 OCR Support: Extract text from images and scanned PDFs using:
Tesseract OCR
NanoNets OCR
Whisper (via llmwhisperer-client)


🧠 AI-Generated Fake Data: Replace sensitive data with realistic fake data using OpenAI.
🌐 Azure Cognitive Services: Fallback support for advanced text analytics.
🧱 Annotated Previews: Visualize redacted text with st-annotated-text.
⚡ Streamlit UI: Fast, interactive, and user-friendly web interface.


🧰 Tech Stack



Category
Tools / Libraries



NLP & Redaction
presidio-analyzer, presidio-anonymizer


UI
streamlit, streamlit-tags, st-annotated-text


OCR
pytesseract, nanonets_ocr, llmwhisperer-client


LLMs
openai, langchain, azure-ai-textanalytics


PDF
PyPDF2, PyMuPDF


Language
Python 3.10 + Poetry



⚙️ Installation
Prerequisites

Python: 3.10 or higher
Poetry: Recommended for dependency management (installation guide)
Tesseract: Installed and added to system PATH (Tesseract installation)

Option 1: Using Poetry (Recommended)
git clone https://github.com/ompatil1906/RedactPlatform.git
cd RedactPlatform
poetry install
poetry shell
streamlit run presidio_streamlit.py

Option 2: Using Pip
git clone https://github.com/ompatil1906/RedactPlatform.git
cd RedactPlatform
pip install -r requirements.txt
streamlit run presidio_streamlit.py


📁 File Structure
RedactPlatform/
├── presidio_streamlit.py             # Main Streamlit application
├── presidio_helpers.py              # Presidio wrapper utilities
├── presidio_nlp_engine_config.py    # Custom NLP configuration
├── openai_fake_data_generator.py    # OpenAI fake data generation logic
├── nanonets_ocr.py                  # NanoNets OCR integration
├── llm_whisper.py                   # Whisper OCR integration
├── requirements.txt                 # Pip-based dependencies
├── pyproject.toml                   # Poetry project configuration
├── poetry.lock                      # Poetry lock file
├── .env                             # API keys and environment variables


🧪 Usage Guide

Run the Application:
streamlit run presidio_streamlit.py


Upload a File: Upload a PDF or text file via the Streamlit interface.

Choose Detection & Redaction Method:

Presidio for standard PII detection
OpenAI LLM for advanced contextual redaction
OCR-based pipeline (Tesseract, NanoNets, or Whisper) for scanned documents


Review and Download: View the redacted output with annotations and download the final file.



🔐 Environment Setup
Create a .env file in the root directory with the following API keys:
OPENAI_API_KEY=your-openai-api-key
NANONETS_API_KEY=your-nanonets-api-key
AZURE_API_KEY=your-azure-text-api-key
AZURE_ENDPOINT=https://your-azure-endpoint.cognitiveservices.azure.com/


Note: Ensure API keys are kept secure and not exposed in version control.


📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgements

Microsoft Presidio
Streamlit
OpenAI API
NanoNets
Azure AI

Made with ❤️ by Om Patil
