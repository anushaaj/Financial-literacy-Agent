# Digital Financial Literacy Agent

A multilingual Streamlit assistant for financial-literacy questions about UPI, online scams, budgeting, and loans. It retrieves grounded guidance from a local ChromaDB knowledge base and multilingual sentence-transformer embeddings. IBM watsonx.ai generation with Granite is implemented as an optional integration: it is used only when an API key and project ID are configured. Without them, the app returns formatted retrieval results.

## Run locally

Use Python 3.11, then install dependencies and start Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model and builds the local vector store from the built-in multilingual guidance. You can optionally add `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, and `WATSONX_MODEL_ID` to a local `.env` file. Never commit credentials.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository, select the `main` branch, set the app file to `app.py`, and choose Python 3.11.
3. In the app's **Settings → Secrets**, add the following TOML, replacing the values with credentials from your IBM watsonx.ai project:

   ```toml
   WATSONX_APIKEY = "your-IBM-Cloud-API-key"
   WATSONX_PROJECT_ID = "your-watsonx-project-id"
   WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
   WATSONX_MODEL_ID = "ibm/granite-3-8b-instruct"
   ```

   Select the correct regional URL for your watsonx.ai deployment. These secrets belong in Streamlit's app settings, not in GitHub files or source control. A local template is available at `.streamlit/secrets.toml.example`.
4. Restart the app after saving secrets. The sidebar reports whether Watsonx credentials are configured. A configured status means the app will attempt Watsonx generation; it does not by itself prove a successful IBM API call. Verify by asking a question and checking that it returns successfully.

The app also starts without Watsonx credentials, using retrieval-only answers. Uploaded documents and the local ChromaDB index are runtime-local and may not persist across hosted app restarts.

## IBM integration

The app calls IBM watsonx.ai's `ibm-watsonx-ai` SDK to generate grounded answers with the configured Granite model when both `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` are available. The integration does not imply use of other IBM products or tools. See [requirements.txt](requirements.txt) for dependencies.

## Project materials

- [IBM project presentation](IBM.pptx)
- [Problem statement](problem.pdf)
- [Architecture blueprint](architecture_blueprint.svg)
- [Application](app.py)
- [RAG engine](rag_engine.py)
- [Tests](test_rag_engine.py)
