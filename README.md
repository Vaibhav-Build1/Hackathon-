# INTELLIGENCE PLUGIN

**Tagline:**  
**Upload a PDF. Understand it. Structure it. Export it.**

This repository contains a local-first Streamlit application for document intelligence on PDFs.

---

## Phase 1 Status (Current)

Implemented in this phase:

- Python-only local application
- Streamlit UI with upload + process flow
- Typed settings via environment variables and YAML
- PDF validation:
  - `.pdf` extension check
  - PDF signature check (`%PDF-`)
  - non-empty file check
  - max file size check
  - openability/corruption check
  - password-protection check
  - max page count check
- PDF inspection:
  - native text density analysis
  - native/scanned/mixed type detection
  - OCR-required pages list
- Native text extraction using PyMuPDF
- OCR adapter with Tesseract (`pytesseract`) and graceful fallback when unavailable
- Processing summary and extraction previews in UI

---

## Local Windows Setup (No Docker, No Git Required)

### 1) Confirm Python version

```powershell
py --version
```

Use Python **3.11+**.

### 2) Create virtual environment

```powershell
py -m venv .venv
```

### 3) Activate environment (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

If blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4) Activate environment (Command Prompt alternative)

```bat
.venv\Scripts\activate.bat
```

### 5) Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 6) Install dependencies

```powershell
pip install -r requirements.txt
```

### 7) Run application

```powershell
streamlit run app.py
```

### 8) Run tests (later phases)

```powershell
pytest
```

---

## OpenRouter AI OCR Fallback (Alternative to Tesseract)

If installing Tesseract is blocked on your machine, you can use **AI Enhanced Mode** with **OpenRouter** instead. This sends only the specific scanned pages (not the whole document) to a vision-capable LLM for OCR.

### 1) Get an OpenRouter API Key
- Create an account at https://openrouter.ai
- Generate an API key from your account dashboard.

### 2) Configure the key
Add to `.env`:
```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TEXT_MODEL=nvidia/nemotron-nano-9b-v2:free
OPENROUTER_VISION_MODEL=qwen/qwen2.5-vl-72b-instruct:free
```

Or paste the key directly into the sidebar field **"OpenRouter API Key (optional)"**.

### 3) Use AI Enhanced Mode
In the sidebar, set **Mode = "AI Enhanced"**. When Tesseract is unavailable, the app will automatically try OpenRouter vision OCR on pages that need OCR.

Notes:
- Local Mode never calls this API. It is only used if you explicitly pick AI Enhanced Mode.
- `nvidia/nemotron-nano-9b-v2:free` is used for general text/classification assistance; vision OCR uses a vision-capable model (default `qwen/qwen2.5-vl-72b-instruct:free`) since Nemotron Nano is text-only. You can change these in `.env`.

---

## Tesseract OCR Setup (Windows)

The app runs even if Tesseract is not installed, but OCR pages will be skipped.

### Option A: Install Tesseract and add to PATH
- Install Tesseract OCR for Windows.
- Ensure `tesseract.exe` is available in PATH.

### Option B: Configure using `.env`
Create a `.env` file (copy from `.env.example`) and set:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Option C: Configure from Streamlit Sidebar
- Use **“Tesseract executable path (optional)”** in the sidebar settings.

---

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

Key values:

- `MAX_UPLOAD_MB=20`
- `MAX_PDF_PAGES=100`
- `MAX_BATCH_FILES=10`
- `OCR_DPI=300`
- `OCR_LANGUAGE=eng`
- `LLM_PROVIDER=disabled` (local mode)

---

## Current Project Files (Phase 1 Core)

- `app.py`
- `.env.example`
- `requirements.txt`
- `src/config/settings.py`
- `src/document_processing/pdf_validator.py`
- `src/document_processing/pdf_inspector.py`
- `src/document_processing/text_extractor.py`
- `src/document_processing/ocr_service.py`

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'yaml'`
Install dependencies:

```powershell
pip install -r requirements.txt
```

### OCR unavailable warning in UI
- Install Tesseract, or
- Set `TESSERACT_CMD` in `.env`, or
- Set path in sidebar input.

### PDF validation failed
Check:
- File is real PDF
- Not empty
- Not password protected
- Under configured size/page limits

---

## Next Phase

Phase 2 adds:
- Document schemas (Pydantic v2)
- Classification (resume/invoice/receipt/others)
- Structured extraction
- Normalization
- Validation
- Confidence and evidence layers
