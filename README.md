Carbon Desk

Flask chat UI for questions about the Indian Carbon Credit Trading Scheme (CCTS). It retrieves context from the existing FAISS vector store, then asks the configured Groq model to answer from those sources.

## Run locally

Install the dependencies and start Flask:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. The UI uses the processed files in `data/` (`chunks.pkl`, `embeddings.npy`, and `faiss.index`).

In the desktop UI, select one or more PDFs with **Upload PDFs**, wait for them to appear in the source list, then select **Ingest PDFs** to rebuild the chunks and vector store. Ask questions separately in the bar at the bottom of the page. Uploading files does not start ingestion automatically.

Set the Groq key locally in an untracked `.env` file before starting the app:

```text
GROQ_API_KEY=your-groq-api-key
```

## Deploy to Vercel

The root `app.py` is the Flask entrypoint. Import this repository as a Vercel project with the project root set to `Carbon_AI_Chatbot`, or install the Vercel CLI and run `vercel` from this directory. Vercel serves assets from `public/`; `vercel.json` includes the vector data and excludes source PDFs and local-only scripts from the function bundle.

Set `GROQ_API_KEY` in the Vercel project environment variables for Production, Preview, and Development as needed. Do not commit `.env`; it is ignored by Git.

### Vercel runtime considerations

The `sentence-transformers`, PyTorch, and FAISS dependencies make this a large Python function. Vercel's standard Python bundle limit is 500 MB; eligible projects can enable Large Functions (up to 5 GB, currently beta) by setting `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the project environment. The embedding model is downloaded from Hugging Face the first time a function instance handles a query, so cold starts can be slow and the model cache is not persistent across instances.

## Ingest documents

The CLI can ingest every PDF in `doc/` or a single PDF:

```powershell
python ingest.py
python ingest.py "path\to\document.pdf"
```

Ingestion writes the combined chunks and embeddings to `data/`. Redeploy after changing those files so the Vercel deployment receives the updated vector store.