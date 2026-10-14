# 🎧 SupportPilot — AI Agent Co-pilot for E-commerce Customer Support

An end-to-end Streamlit app that reads every customer-support ticket, understands it, pulls the order and the
right company policy, recommends the next best action and drafts a reply for the human agent to send.

**Use case source:** ZenML LLMOps Database → E-commerce → *Wayfair: AI-Powered Co-pilot System for Digital
Sales Agents* (reported ~10% reduction in handle time in initial tests).

---

## 1. Run it on your laptop (5 minutes)

You need Python 3.10+ installed.

```bash
# inside the supportpilot folder
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. It works **fully offline** — no API key needed.

(Optional) regenerate the synthetic data: `python generate_data.py`

## 2. Deploy free on Streamlit Community Cloud

1. Create a GitHub account (if you don't have one) and a new **public** repository, e.g. `supportpilot`.
2. Upload all files from this folder (keep the `data/` and `.streamlit/` folders).
   On GitHub: *Add file → Upload files* → drag the folder contents → *Commit*.
3. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app** → *Deploy a public app from GitHub*.
4. Repository: `your-username/supportpilot` · Branch: `main` · Main file path: `app.py` → **Deploy**.
5. Wait 2–3 minutes. You get a public link like `https://supportpilot-xxxx.streamlit.app` — put it on your slides.

## 3. (Optional) Turn on real LLM-written replies

1. Get a free API key at **https://console.groq.com/keys**.
2. Streamlit Cloud → your app → **Settings → Secrets** → paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
3. In the app sidebar choose **Groq**. If the key is missing or the API fails, the app automatically falls back to
   grounded templates, so your live demo can never break.

## 4. Project structure

```
supportpilot/
├── app.py               # Streamlit UI (6 pages)
├── engine.py            # AI pipeline: intent model, sentiment/priority, lookup, RAG, actions, replies, impact model
├── generate_data.py     # synthetic data generator
├── data/                # tickets, orders, customers, products, policies
├── requirements.txt
└── .streamlit/          # theme + secrets example
```

## 5. App pages

| Page | What it shows |
|---|---|
| 🏠 Overview | Problem, solution, headline impact numbers, reference case study |
| 🎧 Agent Workspace | Live demo: ticket → intent, sentiment, priority, order, policy, action, draft reply, approve & send |
| 📥 Smart Ticket Queue | 150 tickets triaged in < 1 s, sorted by urgency; intent and priority mix |
| 💰 Business Impact | Interactive cost model, scenarios (10/25/40%), success criteria & KPIs |
| 📊 Model Evaluation | Accuracy on test split and unseen phrasing, confusion matrix, latency, guardrails |
| 🏗️ How It Works | Architecture diagram, tech stack, path to production, limitations |

All data is synthetic for a fictional retailer, *NovaMart*.
