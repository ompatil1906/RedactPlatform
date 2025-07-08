from typing import List, Optional, Tuple
import logging
import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
import base64
import os
import traceback
import dotenv
import pandas as pd
from annotated_text import annotated_text
from streamlit_tags import st_tags

# Import your helper modules
from openai_fake_data_generator import OpenAIParams
from presidio_helpers import (
    get_supported_entities,
    analyze,
    anonymize,
    annotate,
    create_fake_data,
    analyzer_engine,
)
from llm_whisper import extract_text_from_pdf

# --- PDF Replacement Helper Function ---
def _get_entity_rects(page, result, page_offset):
    words = page.get_text("words")
    spans = []
    page_text = page.get_text()
    entity_span = (result.start - page_offset, result.end - page_offset)
    char_idx = 0
    for w in words:
        word_text = w[4]
        word_start = page_text.find(word_text, char_idx)
        if word_start < 0:
            continue
        word_end = word_start + len(word_text)
        char_idx = word_end
        overlap = not (word_end <= entity_span[0] or word_start >= entity_span[1])
        if overlap:
            spans.append(fitz.Rect(w[:4]))
            print(f"[DEBUG] Matched entity span {entity_span} with word '{word_text}' ({word_start}, {word_end}) -> rect {w[:4]}.")
        else:
            print(f"[DEBUG] No overlap for entity span {entity_span} and word '{word_text}' ({word_start}, {word_end}).")
    if not spans:
        print(f"[DEBUG][MISS] No rect found for entity '{getattr(result,'word',None)}' span={entity_span}.")
    return spans


def _find_textboxes_for_entity(page, entity_text):
    """Find all bounding boxes of literal entity_text matches on the page (case-sensitive)."""
    rects = []
    page_txt = page.get_text()
    for inst in page.search_for(entity_text):
        rects.append(inst)
    return rects


# Ensure st_analyze_results is always defined to prevent NameError for any input type
st_analyze_results = None

def replace_pdf(pdf_document, analyze_results, st_text, replacement_text="<REDACTED>"):
    for page_num, page in enumerate(pdf_document):
        for result in analyze_results:
            try:
                entity_text = result.get_decoded_value(st_text) if hasattr(result, 'get_decoded_value') else st_text[result.start:result.end]
            except Exception:
                entity_text = st_text[result.start:result.end]
            if not entity_text.strip():
                continue
            rects = _find_textboxes_for_entity(page, entity_text)
            if rects:
                for rect in rects:
                    page.add_redact_annot(rect, fill=(1,1,1))
        page.apply_redactions()
        for result in analyze_results:
            try:
                entity_text = result.get_decoded_value(st_text) if hasattr(result, 'get_decoded_value') else st_text[result.start:result.end]
            except Exception:
                entity_text = st_text[result.start:result.end]
            rects = _find_textboxes_for_entity(page, entity_text)
            if rects:
                for rect in rects:
                    page.insert_textbox(rect, replacement_text, fontsize=10, color=(0,0,0), align=1)
    replaced_pdf_bytes = pdf_document.write()
    return replaced_pdf_bytes

def redact_pdf(pdf_document, analyze_results, st_text=None):
    if st_text is None:
        import streamlit as st
        st_text = st.session_state.get('st_text', None)
    if st_text is None:
        raise ValueError("No text context provided for entity mapping (st_text is None)")
    for page_num, page in enumerate(pdf_document):
        for result in analyze_results:
            try:
                entity_text = result.get_decoded_value(st_text) if hasattr(result, 'get_decoded_value') else st_text[result.start:result.end]
            except Exception:
                entity_text = st_text[result.start:result.end]
            if not entity_text.strip():
                continue
            rects = _find_textboxes_for_entity(page, entity_text)
            if rects:
                for rect in rects:
                    page.add_redact_annot(rect, fill=(0,0,0))
        page.apply_redactions()
    redacted_pdf_bytes = pdf_document.write()
    return redacted_pdf_bytes

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="Presidio demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/presidio/",
    },
)

dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")

allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

# Sidebar
st.sidebar.header(
    """
PII De-Identification with [NEXYOM](https://www.nexyom.com)
"""
)

model_help_text = """
    Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
    Presidio supports multiple NER packages, such as Huggingface,
    as well as service such as Azure Text Analytics PII.
    """
st_ta_key = st_ta_endpoint = ""

model_list = [
    
    
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    
    "Azure AI Language",
    "Other",
]
if not allow_other_models:
    model_list.pop()
st_model = st.sidebar.selectbox(
    "NER model package",
    model_list,
    index=0,
    help=model_help_text,
)

st_model_package = st_model.split("/")[0]
st_model = (
    st_model
    if st_model_package.lower() not in ("huggingface")
    else "/".join(st_model.split("/")[1:])
)

if st_model == "Other":
    st_model_package = st.sidebar.selectbox(
        "NER model OSS package", options=["HuggingFace"]
    )
    st_model = st.sidebar.text_input(f"NER model name", value="")

if st_model == "Azure AI Language":
    st_ta_key = st.sidebar.text_input(
        f"Azure AI Language key", value=os.getenv("TA_KEY", ""), type="password"
    )
    st_ta_endpoint = st.sidebar.text_input(
        f"Azure AI Language endpoint",
        value=os.getenv("TA_ENDPOINT", default=""),
        help="For more info: https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/personally-identifiable-information/overview",
    )

st.sidebar.warning("Note: Models might take some time to download. ")

analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)
logger.debug(f"analyzer_params: {analyzer_params}")

st_operator = st.sidebar.selectbox(
    "De-identification approach",
    ["redact", "replace", "synthesize", "highlight", "mask", "hash", "encrypt"],
    index=1,
    help="""
    Select which manipulation to the text is requested after PII has been identified.\n
    - Redact: Completely remove the PII text\n
    - Replace: Replace the PII text with a constant, e.g. <PERSON>\n
    - Synthesize: Replace with fake values (requires an OpenAI key)\n
    - Highlight: Shows the original text with PII highlighted in colors\n
    - Mask: Replaces a requested number of characters with an asterisk (or other mask character)\n
    - Hash: Replaces with the hash of the PII string\n
    - Encrypt: Replaces with an AES encryption of the PII string, allowing the process to be reversed
         """,
)
st_mask_char = "*"
st_number_of_chars = 15
st_encrypt_key = "WmZq4t7w!z%C&F)J"

open_ai_params = None

def set_up_openai_synthesis():
    if os.getenv("OPENAI_TYPE", default="openai") == "Azure":
        openai_api_type = "azure"
        st_openai_api_base = st.sidebar.text_input(
            "Azure OpenAI base URL",
            value=os.getenv("AZURE_OPENAI_ENDPOINT", default=""),
        )
        openai_key = os.getenv("AZURE_OPENAI_KEY", default="")
        st_deployment_id = st.sidebar.text_input(
            "Deployment name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", default="")
        )
        st_openai_version = st.sidebar.text_input(
            "OpenAI version",
            value=os.getenv("OPENAI_API_VERSION", default="2023-05-15"),
        )
    else:
        openai_api_type = "openai"
        st_openai_version = st_openai_api_base = None
        st_deployment_id = ""
        openai_key = os.getenv("OPENAI_KEY", default="")
    st_openai_key = st.sidebar.text_input(
        "OPENAI_KEY",
        value=openai_key,
        help="See https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key for more info.",
        type="password",
    )
    st_openai_model = st.sidebar.text_input(
        "OpenAI model for text synthesis",
        value=os.getenv("OPENAI_MODEL", default="gpt-3.5-turbo-instruct"),
        help="See more here: https://platform.openai.com/docs/models/",
    )
    return (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    )

if st_operator == "mask":
    st_number_of_chars = st.sidebar.number_input(
        "number of chars", value=st_number_of_chars, min_value=0, max_value=100
    )
    st_mask_char = st.sidebar.text_input(
        "Mask character", value=st_mask_char, max_chars=1
    )
elif st_operator == "encrypt":
    st_encrypt_key = st.sidebar.text_input("AES key", value=st_encrypt_key)
elif st_operator == "synthesize":
    (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    ) = set_up_openai_synthesis()

    open_ai_params = OpenAIParams(
        openai_key=st_openai_key,
        model=st_openai_model,
        api_base=st_openai_api_base,
        deployment_id=st_deployment_id,
        api_version=st_openai_version,
        api_type=openai_api_type,
    )

st_threshold = st.sidebar.slider(
    label="Acceptance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    help="Define the threshold for accepting a detection as PII. See more here: ",
)

st_return_decision_process = st.sidebar.checkbox(
    "Add analysis explanations to findings",
    value=False,
    help="Add the decision process to the output table. "
    "More information can be found here: https://microsoft.github.io/presidio/analyzer/decision_process/",
)

# Allow and deny lists
st_deny_allow_expander = st.sidebar.expander(
    "Allowlists and denylists",
    expanded=False,
)

with st_deny_allow_expander:
    st_allow_list = st_tags(
        label="Add words to the allowlist", text="Enter word and press enter."
    )
    st.caption(
        "Allowlists contain words that are not considered PII, but are detected as such."
    )

    st_deny_list = st_tags(
        label="Add words to the denylist", text="Enter word and press enter."
    )
    st.caption(
        "Denylists contain words that are considered PII, but are not detected as such."
    )

# Main panel
with st.expander("About this demo", expanded=False):


    st.info(
        """
    Use this demo to:
    - Experiment with different off-the-shelf models and NLP packages.
    - Explore the different de-identification options, including redaction, masking, encryption and more.
    - Generate synthetic text with Microsoft Presidio and OpenAI.
    - Configure allow and deny lists.
    
    This demo website shows some of Presidio's capabilities.
    [Visit our website](https://www.nexyom.com) for more info,
    samples and deployment options.    
    """
    )

    

analyzer_load_state = st.info("Starting Presidio analyzer...")
analyzer_load_state.empty()

# Read default text
with open("demo_text.txt") as f:
    demo_text = f.readlines()

# Create two columns for before and after
col1, col2 = st.columns(2)

# Choose entities (must be available before any analysis logic)
st_entities_expander = st.sidebar.expander("Choose entities to look for")
st_entities = st_entities_expander.multiselect(
    label="Which entities to look for?",
    options=get_supported_entities(*analyzer_params),
    default=list(get_supported_entities(*analyzer_params)),
    help="Limit the list of PII entities detected. "
         "This list is dynamic and based on the NER model and registered recognizers. "
         "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
)
# Before:
col1.subheader("Input")
input_type = col1.radio("Input type", ["Text", "PDF", "Image"])

st_text = ""
pdf_document = None
uploaded_file = None
img = None
img_bytes = None
if input_type == "Text":
    st_text = col1.text_area(
        label="Enter text", value="".join(demo_text), height=400, key="text_input"
    )
elif input_type == "PDF":
    # --- LLM Whisperer OCR Integration ---
    from llm_whisper import extract_text_from_pdf
    uploaded_file = col1.file_uploader("Upload PDF", type="pdf")
    ocr_backend = col1.selectbox("OCR backend for PDF", ["Tesseract", "Surya", "LLM Whisperer"], index=0)
    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        import fitz  # Safe to import here
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if ocr_backend == "LLM Whisperer":
            st_text = extract_text_from_pdf(temp_pdf_path)
        else:
            # Default to Tesseract or Surya, depending on further expansion.
            st_text = "\n".join(page.get_text() for page in pdf_document)
    else:
        st.info("Please upload a PDF file.")
        st.stop()
elif input_type == "Image":
    uploaded_file = col1.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    from nanonets_ocr import ocr_page_with_nanonets_s
    ocr_backend = col1.selectbox("OCR backend for Image", ["Tesseract", "LLM Whisperer", "Nanonets OCR"], index=0)
    if uploaded_file:
        from PIL import Image, ImageDraw
        img_bytes = uploaded_file.read()
        img = Image.open(uploaded_file)
        temp_img_path = "temp_uploaded_image.png"
        img.save(temp_img_path)
        if ocr_backend == "LLM Whisperer":
            st_text = extract_text_from_pdf(temp_img_path)
            # Handle dict response from LLM Whisperer
            if isinstance(st_text, dict):
                extraction = st_text.get("extraction") or st_text.get("extracted_text")
                if isinstance(extraction, dict) and "confidence_metadata" in extraction:
                    # Flatten all text segments in order
                    all_lines = []
                    for block in extraction["confidence_metadata"]:
                        line = " ".join([seg.get("text", "") for seg in block if isinstance(seg, dict) and "text" in seg])
                        if line.strip():
                            all_lines.append(line.strip())
                    st_text = "\n".join(all_lines).strip()
                elif isinstance(extraction, str):
                    st_text = extraction
                # fallback: avoid showing raw dict in UI
                elif not isinstance(st_text, str):
                    st_text = ""
            st.markdown("### Extracted Text (LLM Whisperer)")
            st.code(st_text or "No text extracted", language="text")
        elif ocr_backend == "Nanonets OCR":
            # --- Nanonets Model Session State ---
            if "nanonets_model" not in st.session_state or "nanonets_processor" not in st.session_state:
                st.session_state["nanonets_model"], st.session_state["nanonets_processor"] = None, None
            
            def get_nanonets_model():
                if st.session_state["nanonets_model"] is None or st.session_state["nanonets_processor"] is None:
                    from nanonets_ocr import load_nanonets_model
                    model, processor = load_nanonets_model()
                    st.session_state["nanonets_model"] = model
                    st.session_state["nanonets_processor"] = processor
                return st.session_state["nanonets_model"], st.session_state["nanonets_processor"]
            from nanonets_ocr import ocr_page_with_nanonets_s
            model, processor = get_nanonets_model()
            st_text = ocr_page_with_nanonets_s(temp_img_path, model=model, processor=processor)
            st.markdown("### Extracted Text (Nanonets OCR)")
            st.code(st_text or "No text extracted", language="text")
        else:
            import pytesseract
            st_text = pytesseract.image_to_string(img)
            st.markdown("### Extracted Text (Tesseract)")
            st.code(st_text or "No text extracted", language="text")
        # rest of OCR/entity extraction logic proceeds...
    else:
        st.info("Please upload an image file.")
        st.stop()

analyzer_load_state = st.info("Starting Presidio analyzer...")
analyzer = analyzer_engine(*analyzer_params)
try:    
    analyzer_load_state.empty()
    if input_type == "Image":
        st_analyze_results = []
    else:
        st_analyze_results = analyze(
            *analyzer_params,
            text=st_text,
            entities=st_entities,
            language="en",
            score_threshold=st_threshold,
            return_decision_process=st_return_decision_process,
            allow_list=st_allow_list,
            deny_list=st_deny_list,
        )

    # After
    # Use st.columns with spacing to align and separate the PDFs
    pdf_cols = st.columns([1, 0.04, 1])
    if input_type == "Text":
        with col2:
            st.subheader("Output")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                mask_char=st_mask_char,
                number_of_chars=st_number_of_chars,
                encrypt_key=st_encrypt_key,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="De-identified", value=st_anonymize_results.text, height=400
            )
    if input_type == "PDF" and uploaded_file and st_operator == "redact":
        with pdf_cols[0]:
            st.subheader("Original PDF Preview")
            if uploaded_file:
                base64_pdf_orig = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_display_orig = f'<iframe src="data:application/pdf;base64,{base64_pdf_orig}" width="100%" height="900" type="application/pdf" style="border:1px solid #eee;"></iframe>'
                st.markdown(pdf_display_orig, unsafe_allow_html=True)
                st.download_button(
                    label="Download original PDF",
                    data=pdf_bytes,
                    file_name="original_document.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("Please upload a PDF file.")
        with pdf_cols[2]:
            st.subheader("Redacted PDF Preview")
            if pdf_document:
                redacted_pdf_bytes = redact_pdf(pdf_document, st_analyze_results, st_text)
                base64_pdf = base64.b64encode(redacted_pdf_bytes).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf" style="border:1px solid #eee;"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.download_button(
                    label="Download redacted PDF",
                    data=redacted_pdf_bytes,
                    file_name="redacted_document.pdf",
                    mime="application/pdf"
                )
                # Show Findings for Redact in Table Format
                st.markdown("**Findings**")
                if st_analyze_results:
                    import pandas as pd
                    redact_data = []
                    for res in st_analyze_results:
                        entity_type = getattr(res, 'entity_type', getattr(res, 'entity', ''))
                        text = getattr(res, 'original_text', getattr(res, 'text', ''))
                        redact_data.append({"Entity": entity_type, "Text": text})
                    if redact_data:
                        df = pd.DataFrame(redact_data)
                        st.table(df)
                    else:
                        st.text("No findings")
                else:
                    st.text("No findings")
    elif input_type == "PDF" and uploaded_file and st_operator == "replace":
        with pdf_cols[0]:
            st.subheader("Original PDF Preview")
            if uploaded_file:
                base64_pdf_orig = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_display_orig = f'<iframe src="data:application/pdf;base64,{base64_pdf_orig}" width="100%" height="900" type="application/pdf" style="border:1px solid #eee;"></iframe>'
                st.markdown(pdf_display_orig, unsafe_allow_html=True)
                st.download_button(
                    label="Download original PDF",
                    data=pdf_bytes,
                    file_name="original_document.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("Please upload a PDF file.")
        with pdf_cols[2]:
            st.subheader("PII-Replaced PDF Preview")
            if pdf_document:
                replaced_pdf_bytes = replace_pdf(pdf_document, st_analyze_results, st_text, replacement_text="<REDACTED>")
                base64_pdf = base64.b64encode(replaced_pdf_bytes).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf" style="border:1px solid #eee;"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.download_button(
                    label="Download PII-Replaced PDF",
                    data=replaced_pdf_bytes,
                    file_name="replaced_document.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("Please upload a PDF file to replace sensitive data.")
    elif input_type == "PDF" and st_operator not in ("highlight", "synthesize"):
        with col2:
            st.subheader("Output")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                mask_char=st_mask_char,
                number_of_chars=st_number_of_chars,
                encrypt_key=st_encrypt_key,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="De-identified", value=st_anonymize_results.text, height=400
            )
    elif input_type == "PDF" and st_operator == "synthesize":
        with col2:
            st.subheader("OpenAI Generated output")
            fake_data = create_fake_data(
                st_text,
                st_analyze_results,
                open_ai_params,
            )
            st.text_area(label="Synthetic data", value=fake_data, height=400)
        

    # table result
    if not (input_type == "PDF" and uploaded_file and st_operator == "redact"):
        st.subheader(
            "Findings"
            if not st_return_decision_process
            else "Findings with decision factors"
        )
        if st_analyze_results:
            df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
            df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]
            df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
                {
                    "entity_type": "Entity type",
                    "text": "Text",
                    "start": "Start",
                    "end": "End",
                    "score": "Confidence",
                },
                axis=1,
            )
            df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
            if st_return_decision_process:
                analysis_explanation_df = pd.DataFrame.from_records(
                    [r.analysis_explanation.to_dict() for r in st_analyze_results]
                )
                df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
            st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
        else:
            st.text("No findings")

except Exception as e:
    print(e)
    traceback.print_exc()
    st.error(e)

components.html(
    """
    <script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "h7f8bp42n8");
    </script>
    """
)


if input_type == "Image" and uploaded_file and (st_operator == "redact" or st_operator == "replace"):
    pdf_cols = st.columns([1, 0.04, 1])
    with pdf_cols[0]:
        st.subheader("Original Image Preview")
        st.image(img, use_container_width=True)
        if img_bytes:
            st.download_button(
                label="Download original image",
                data=img_bytes,
                file_name="original_image.png",
                mime="image/png",
            )
        else:
            st.warning("Please upload an image file.")
    with pdf_cols[2]:
        if st_operator == "redact":
            st.subheader("Redacted Image Preview")
        else:
            st.subheader("PII-Replaced Image Preview")
        if img:
            # Ensure image mode supports alpha for visible filled rectangles
            if img.mode != "RGBA":
                img_processed = img.convert("RGBA")
            else:
                img_processed = img.copy()
            draw = ImageDraw.Draw(img_processed)
            if st_analyze_results:
                for result in st_analyze_results:
                    if hasattr(result, 'bbox') and result.bbox:
                        x0, y0, x1, y1 = result.bbox
                        if st_operator == "redact":
                            box_color = (0, 0, 0, 255)  # Opaque black
                        else:
                            box_color = (255, 255, 255, 255)  # Opaque white
                        # Draw a filled and outlined rectangle for visual clarity
                        draw.rectangle([x0, y0, x1, y1], fill=box_color, outline="red", width=2)
            st.image(img_processed, use_container_width=True)
            import io
            buf = io.BytesIO()
            img_processed.save(buf, format="PNG")
            st.download_button(
                label=f"Download { 'redacted' if st_operator == 'redact' else 'replaced' } image",
                data=buf.getvalue(),
                file_name=f"{ 'redacted' if st_operator == 'redact' else 'replaced' }_image.png",
                mime="image/png",
            )
            # Show Findings for Redact
            st.markdown("**Findings**")
            if st_analyze_results:
                for result in st_analyze_results:
                    st.write(result)
            else:
                st.text("No findings")
        else:
            st.warning("Please upload an image file to process.")

            
# --- Image Replace ---
elif input_type == "Image" and uploaded_file and st_operator == "replace":
    pdf_cols = st.columns([1, 0.04, 1])
    with pdf_cols[0]:
        st.subheader("Original Image Preview")
        st.image(img, use_container_width=True)
        if img_bytes:
            st.download_button(
                label="Download original image",
                data=img_bytes,
                file_name="original_image.png",
                mime="image/png",
            )
        else:
            st.warning("Please upload an image file.")
    with pdf_cols[2]:
        st.subheader("PII-Replaced Image Preview")
        if img:
            img_replaced = img.copy()
            draw = ImageDraw.Draw(img_replaced)
            if st_analyze_results:
                for result in st_analyze_results:
                    if hasattr(result, 'bbox') and result.bbox:
                        x0, y0, x1, y1 = result.bbox
                        draw.rectangle([x0, y0, x1, y1], fill="white")
            else:
                w, h = img_replaced.size
                draw.rectangle([w//4, h//4, w*3//4, h//2], fill="white")
            st.image(img_replaced, use_container_width=True)
            import io
            buf = io.BytesIO()
            img_replaced.save(buf, format="PNG")
            st.download_button(
                label="Download replaced image",
                data=buf.getvalue(),
                file_name="replaced_image.png",
                mime="image/png",
            )
        else:
            st.warning("Please upload an image file to replace sensitive data.")
    
    if st_operator == "synthesize":
        with col2:
            st.subheader("OpenAI Generated output")
            fake_data = create_fake_data(
                st_text,
                st_analyze_results,
                open_ai_params,
            )
            st.text_area(label="Synthetic data", value=fake_data, height=400)
    else:
        st.subheader("Highlighted")
        annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

    # table result
    if not (input_type == "PDF" and uploaded_file and st_operator == "redact"):
        st.subheader(
            "Findings"
            if not st_return_decision_process
            else "Findings with decision factors"
        )
        if st_analyze_results:
            df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
            df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]
            df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
                {
                    "entity_type": "Entity type",
                    "text": "Text",
                    "start": "Start",
                    "end": "End",
                    "score": "Confidence",
                },
                axis=1,
            )
            df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
            if st_return_decision_process:
                analysis_explanation_df = pd.DataFrame.from_records(
                    [r.analysis_explanation.to_dict() for r in st_analyze_results]
                )
                df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
            st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
        else:
            st.text("No findings")


components.html(
    """
    <script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "h7f8bp42n8");
    </script>
    """
)
