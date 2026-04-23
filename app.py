import os
import io
import re
import gradio as gr
from contextlib import redirect_stdout
from main import run_pipeline

def clean_output(text):
    """
    Cleans raw pipeline output by removing debug logs, system timing, 
    and redundant traces to present a user-facing financial briefing.
    """
    lines = text.split('\n')
    cleaned = []
    
    # Noise patterns to exclude
    noise_patterns = [
        r"^\[RULE CHECK INPUT\]",
        r"^\[EVAL\]",
        r"^\[LANGFUSE\]",
        r"^\[PHASE\]",
        r"^\d{4}-\d{2}-\d{2}",  # Standard timestamps
        r"^Starting Autonomous Financial Advisor Agent",
        r"^Analysis Complete",
        r"^\s*$" # Empty lines (we handle spacing later)
    ]
    
    for line in lines:
        if any(re.match(p, line) for p in noise_patterns):
            continue
        
        # Style transformations
        l = line.strip()
        if "─" in l or "═" in l:
            cleaned.append("---")
        elif "[FINAL ADVISORY EXPLANATION]" in l:
            cleaned.append("## 📝 Strategic Advisory Briefing")
        elif "Top Drivers:" in l:
            cleaned.append("### 🚀 Primary Market Drivers")
        elif "Risks & Anomalies:" in l:
            cleaned.append("### ⚠️ Risk Assessment")
        elif "[SYSTEM CONFIDENCE]" in l:
            cleaned.append(f"**{l}**")
        elif "[AI JUDGE SCORE]" in l:
            cleaned.append(f"**{l}**")
        elif "📊" in l:
            cleaned.append(f"# {l}")
        elif l.startswith("- ") or l.startswith("* "):
            cleaned.append(line) # Keep bullet points
        else:
            cleaned.append(line)
            
    return "\n\n".join(cleaned)

def run_analysis(selected_portfolio):
    """
    Executes the AlphaTrace pipeline and applies aesthetic filtering 
    to the results before rendering in the UI.
    """
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            if selected_portfolio == "ALL":
                portfolio_list = ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"]
            else:
                portfolio_list = [selected_portfolio]
            
            run_pipeline(portfolio_list)
        
        raw_output = f.getvalue()
        if not raw_output:
            return "Analysis complete. No output captured."
        
        return clean_output(raw_output)
    except Exception as e:
        return f"### ❌ Error during analysis\n{str(e)}"

# Define Refined Gradio UI
with gr.Blocks(title="AlphaTrace | Causal Intelligence") as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown("# 🔍 AlphaTrace: Causal Reasoning Engine")
        gr.Markdown(
            "Bridging raw market volatility and human-readable insights using deterministic "
            "causal pipelines and AI-driven synthesis."
        )
        
        with gr.Row():
            portfolio_selector = gr.Dropdown(
                choices=["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003", "ALL"],
                value="PORTFOLIO_001",
                label="Select Portfolio Entity"
            )
            run_btn = gr.Button("Run Reasoning Cycle", variant="primary")
        
        gr.Separator()
        
        output_display = gr.Markdown(
            label="Analysis Briefing",
            value="*Results will appear here after starting the reasoning cycle.*"
        )
        
        gr.Separator()
        
        with gr.Accordion("System Information", open=False):
            gr.Markdown(
                "**Architecture:** Multi-stage deterministic pipeline  \n"
                "**Engine:** Llama-3.3-70B (Groq)  \n"
                "**Observability:** Langfuse Tracing Enabled"
            )

    run_btn.click(
        fn=run_analysis,
        inputs=[portfolio_selector],
        outputs=[output_display]
    )

if __name__ == "__main__":
    demo.launch()
