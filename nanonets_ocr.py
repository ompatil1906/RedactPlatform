import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import tempfile

def preprocess_image_for_nanonets(image_path):
    """
    1. Convert .png (or any format) to .jpg
    2. Resize so that height <= 1080px (keep aspect ratio)
    3. Compress as JPEG with quality 80
    Returns the path to the processed image.
    """
    image = Image.open(image_path).convert("RGB")
    # Resize
    h = image.height
    w = image.width
    if h > 1080:
        new_h = 1080
        new_w = int(image.width * (1080 / h))
        image = image.resize((new_w, new_h), Image.LANCZOS)
    # Save as JPEG in a temp file
    _, temp_jpg = tempfile.mkstemp(suffix=".jpg")
    image.save(temp_jpg, "JPEG", quality=80, optimize=True)
    return temp_jpg

def load_nanonets_model(model_path="nanonets/Nanonets-OCR-s"):
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def ocr_page_with_nanonets_s(image_path, model=None, processor=None, max_new_tokens=4096):
    """
    Runs OCR on an image using Nanonets-OCR-s via Hugging Face Transformers.
    Applies preprocessing: converts to JPG, resizes (max height 1080px), and compresses (quality 80).
    """
    # Preprocess image
    processed_path = preprocess_image_for_nanonets(image_path)
    if model is None or processor is None:
        model, processor = load_nanonets_model()
    prompt = ("""Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. """
              "If there is an image in the document and image caption is not present, add a small description inside <img></img>. "
              "Watermarks should be wrapped in <watermark>...</watermark>. "
              "Page numbers should be wrapped in <page_number>...</page_number>. "
              "Prefer using ☐ and ☑ for check boxes.")
    image = Image.open(processed_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{processed_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]