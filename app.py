import os
import io
import gradio as gr
from contextlib import redirect_stdout
from main import run_pipeline

def run_analysis(selected_portfolio):
    """
    Executes the AlphaTrace pipeline for a selected portfolio or all of them.
    Captures console output using StringIO and returns it to the interface.
    """
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            if selected_portfolio == "ALL":
                portfolio_list = ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"]
            else:
                portfolio_list = [selected_portfolio]
            
            run_pipeline(portfolio_list)
        
        output = f.getvalue()
        return output if output else "Analysis complete. Check logs if no output is visible."
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Define Gradio Blocks UI
with gr.Blocks(title="AlphaTrace — Autonomous Financial Reasoning Engine") as demo:
    gr.Markdown("# AlphaTrace — Autonomous Financial Reasoning Engine")
    gr.Markdown(
        "AlphaTrace is a deterministic financial reasoning engine that explains portfolio movements by linking "
        "macroeconomic news, sector trends, and individual stock performance."
    )
    
    with gr.Row():
        portfolio_selector = gr.Dropdown(
            choices=["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003", "ALL"],
            value="PORTFOLIO_001",
            label="Select Portfolio for Analysis"
        )
        run_btn = gr.Button("Run Analysis", variant="primary")
    
    output_display = gr.Textbox(
        label="Pipeline Output",
        placeholder="Analysis results will appear here...",
        lines=30,
        interactive=False
    )
    
    run_btn.click(
        fn=run_analysis,
        inputs=[portfolio_selector],
        outputs=[output_display]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        "**Note:** Ensure `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` are configured "
        "as environment variables for full reasoning and observability support."
    )

if __name__ == "__main__":
    demo.launch()
