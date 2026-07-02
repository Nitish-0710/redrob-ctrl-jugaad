# Streamlit Sandbox Deployment Guide
**Team**: Ctrl + Jugaad  
**App Entrypoint**: `app.py`  
**Dependencies**: `requirements.txt`  

This guide provides simple, step-by-step instructions to deploy the candidate discovery and ranking engine sandbox on **Streamlit Community Cloud**.

---

## 1. Prerequisites
Before deploying, ensure you have:
1. A GitHub repository containing the latest codebase (committed with our root `app.py`, `requirements.txt`, `src/`, `configs/`, etc.).
2. A free account on [Streamlit Community Cloud](https://share.streamlit.io/).
3. A small test dataset (such as `sample_candidates.json` or a subset of `candidates.jsonl`) ready on your local machine to upload.

---

## 2. Step-by-Step Deployment

1. **Sign In**: Navigate to [Streamlit Community Cloud](https://share.streamlit.io/) and log in using your GitHub account.
2. **Create Space**: Click on the **"New app"** button in your Streamlit dashboard.
3. **Repository Details**:
   - **Repository**: Choose `Nitish-0710/redrob-ctrl-jugaad`.
   - **Branch**: Set to `main` (or the default submission branch).
   - **Main file path**: Set to `app.py`.
   - **App URL**: Choose a custom subdomain, e.g., `redrob-ctrl-jugaad.streamlit.app`.
4. **Deploy**: Click the **"Deploy!"** button.
5. **App Provisioning**: Streamlit will provision a container, clone the repository, and automatically install the dependencies declared in `requirements.txt` (including `streamlit`, `pandas`, `pyarrow`, `psutil`, etc.).

---

## 3. Local Verification (Optional)
To verify the application runs locally before checking the hosted cloud instance, run:

```bash
# Activate your environment
source venv/bin/activate

# Execute Streamlit run
streamlit run app.py
```

Streamlit will open the interface in your default web browser (typically at `http://localhost:8501`).

---

## 4. Sandbox Usage Guidelines
Once the application is live:
1. **Upload Dataset**: Drag-and-drop or upload `sample_candidates.json` (or a smaller slice of `candidates.jsonl`) into the file uploader.
2. **Execute Ranking**: The app will automatically run the candidate scoring and ranking pipeline.
3. **Inspect Output**: View the interactive data table showing candidate IDs, ranks, final scores, confidence categories, and recruiter explanations.
4. **Download CSV**: Click the **"Download submission.csv"** button to download the compliant output file directly to your local downloads folder.
