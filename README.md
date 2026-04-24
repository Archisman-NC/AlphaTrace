# AlphaTrace 📈
### **Autonomous Financial Reasoning & Monitoring Copilot**

AlphaTrace is a production-grade AI financial advisor designed to move beyond simple chat. It executes 
**deterministic causal reasoning** to link global market signals directly to portfolio impact.

[**🌐 Live Demo**](https://alphatrace.streamlit.app/) | [**📊 Architecture Documentation**](#architecture)

---

## **The AlphaTrace Difference**
Most AI financial tools are just wrappers around LLMs. AlphaTrace is a **reasoning engine**.

*   **Deterministic Chains**: Links Macro News → Sector Trends → Stock Performance → Portfolio Impact.
*   **Proactive Monitoring**: An autonomous "Active Lookout" (Astor-style) that flags concentration risks and asset-sector divergence without being asked.
*   **Hybrid Intelligence**: Orchestrates Groq (Ultra-low latency reasoning) and OpenAI (Premium narrative synthesis).
*   **Self-Correction**: An internal "Auditor" grades every response on ticker accuracy and causal integrity, triggering a re-generation if quality falls below 6.0/10.

---

## **Core Capabilities**
*   **Multi-Intent Reasoning**: Context-aware understanding of "Why", "Risk", and "Full Analysis" queries.
*   **Proactive Insights**: Autonomous signal detection with interactive follow-up hooks.
*   **Memory-Aware**: High-fidelity structured memory system for multi-turn temporal analysis.
*   **Visual Dashboard**: Real-time Plotly sector analytics and styled risk-performance tables.
*   **Observability**: Full-stack Langfuse instrumentation for tracing, latency monitoring, and quality scoring.

---

## **Architecture Overview**
AlphaTrace utilizes a modular 8-phase reasoning pipeline:

1.  **Context Resolver**: Disambiguates pronouns and resolves portfolio context using episodic memory.
2.  **Intent Classifier**: Maps queries to specific analytical engines (Reason, Risk, Performance).
3.  **Execution Router**: Orchestrates deterministic analytical tools (mock/live financial data).
4.  **Causal Chain Builder**: Extracts triggers from raw news and maps them to asset performance.
5.  **Response Generator**: Syntheses narrative using prioritized memory context.
6.  **Self-Evaluator**: Internal Llama-3.3 auditor audits the draft for tickers, %, and causes.
7.  **Self-Correction Loop**: Automatically regenerates responses if the audit score is insufficient.
8.  **Proactive Engine**: Scans analytical outputs for hidden signals to suggest next-turn analysis.

---

## **Tech Stack**
*   **Logic Engine**: Llama-3.1 & 3.3 (via Groq Cloud)
*   **Narrative Polisher**: GPT-4o (via OpenAI)
*   **Frontend**: Streamlit
*   **Observability**: Langfuse
*   **Analytics**: Pandas, Plotly, Regex Causal Extraction

---

## **Getting Started**

### **1. Clone & Install**
```bash
git clone https://github.com/Archisman-NC/AlphaTrace.git
cd AlphaTrace
pip install -r requirements.txt
```

### **2. Configure Environment**
Create a `.env` file with your credentials:
```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_HOST=your_host
```

### **3. Run Application**
```bash
streamlit run app.py
```

---

## **Try These Queries**
*   **Casual Analysis**: *"Why is my portfolio down today?"*
*   **Risk Deep-Dive**: *"Detect concentration risk in my holdings."*
*   **Temporal Reasoning**: *"Is the risk profile getting worse compared to our last check?"*
*   **Proactive Analysis**: Click the **"🔍 Analyze this signal"** button when the Copilot flags a market divergence.

---

## **Operational Integrity**
AlphaTrace employs a **Strict Scoring Rubric** (0-10) for all reasoning:
*   **Ticker Specicity**: Validated matching against a high-fidelity ticker hub.
*   **Quantification**: Mandatory percentage and P&L grounding.
*   **Causal Trigger**: Exact matching of news-to-impact triggers.

---
*Built for the next generation of autonomous investment management.*
