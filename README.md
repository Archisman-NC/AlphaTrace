# AlphaTrace 📈
### **Autonomous Financial Intelligence & Reasoning Copilot**

AlphaTrace is a production-grade AI financial advisor that moves beyond simple chat. It executes **deterministic causal reasoning** to link global market signals directly to portfolio impact, providing sharp, actionable insights without the fluff.

[**🌐 Live Production App**](https://alphatrace-w7zqnrcq5wcjv4fb4qncrg.streamlit.app/) | [**📺 Video Demo (Loom)**](https://www.loom.com/share/2ebc198b32114ea1a0f17937952aa862) | [**📊 Architecture Documentation**](#architecture)

---

## **The AlphaTrace Difference**
Most AI financial tools are mere wrappers around LLMs. AlphaTrace is a **multi-component reasoning engine** built for institutional-grade stability and precision.

*   🛡️ **Deterministic Integrity**: Every insight is anchored in pre-computed mathematical signals. No hallucinated gains or tickers.
*   🧠 **Adaptive Narrative Synthesis**: Natural, question-driven responses that adapt their structure based on user intent (Analysis vs. Explanation vs. Advice).
*   🛰️ **Proactive Intelligence**: An autonomous "Active Lookout" that flags sector divergence and concentration risks before you ask.
*   ⚖️ **Self-Correcting Loop**: An internal auditor grades every analytical turn on data fidelity, triggering immediate regeneration if the "Analyst Grade" falls below threshold.

---

## **Core Analytical Engines**
*   **Contextual Resolver**: Uses episodic memory to disambiguate complex multi-turn financial queries.
*   **Deterministic Causal Chains**: Extends Macro News → Sector Trends → Asset Performance → Portfolio Impact.
*   **Institutional Sector Profiler**: Real-time exposure mapping across direct stock and indirect mutual fund holdings.
*   **Fidelity Registry**: Dynamic confidence marker surfaced as High/Moderate/Best-Effort signal strength.

---

## **Architecture Ovenview**
AlphaTrace utilizes a modular 8-phase reasoning pipeline:

1.  **Intent Classifier**: Maps queries to specialized analytical engines (Reason, Risk, Comparison).
2.  **Execution Router**: Orchestrates deterministic tools across market and portfolio data sources.
3.  **Causal Linker**: Extracts triggers from news and maps them to quantitative asset performance.
4.  **Narrative Generator**: Synthesizes sharp, analyst-persona responses lead by the specific user question.
5.  **Heuristic Evaluator**: Audits the draft for specific tickers, numeric grounding, and causal logic.
6.  **Proactive Engine**: Scans turn outputs for "hidden signals" to suggest the next logical analysis.

---

## **Tech Stack**
*   **Reasoning Core**: Groq Llama-3.3-70b-versatile (Ultra-low latency intelligence)
*   **Interface**: Streamlit (Premium financial dashboard)
*   **Observability**: Langfuse (Full-stack trace & quality scoring)
*   **Mathematical Core**: Pandas, Plotly, Custom Causal Extraction Parsers
*   **Resilience**: Institutional safe-parsing (`safe_float`) and stable namespace sentinels.

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
streamlit run main.py
```

---

## **Sample Queries**
*   **Contextual Analysis**: *"Why is my portfolio down today compared to the Nifty 50?"*
*   **Strategic Comparison**: *"Compare my IT holdings vs. Banking sector performance."*
*   **Direct Advice**: *"What should I do about the concentration risk in my diversified portfolio?"*

---
*Built for the next generation of autonomous investment management.*
