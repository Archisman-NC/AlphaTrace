import os
import json
import logging
import time
from groq import Groq
from dotenv import load_dotenv
from app.utils.helpers import langfuse

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

STRICT_EVALUATOR_PROMPT = """
You are a strict financial reasoning evaluator.
Your job is to score reasoning quality from 0–10.

CRITERIA:
- Correct causal chain: Does the logic follow established market mechanics?
- No hallucination: Are all facts and metrics present in the raw tool data?
- Clear linkage: Is there a coherent path from NEWS -> SECTOR -> STOCK -> PORTFOLIO?

RULES:
- Be harsh.
- Avoid scores above 9 unless exceptional.
- Downgrade heavily for generic explanations or missed causal links.
- Look for "Why" rather than just "What."

Return ONLY valid JSON:
{
  "score": float,
  "reason": "short explanation of the score (max 20 words)"
}
"""

def evaluate_reasoning_quality(narration: str, chains: list, tools: dict) -> dict:
    """
    Strictly audits the generated narrative against raw causal data.
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        logger.error(f"Evaluator Init Failed: {e}")
        return {"score": 0.0, "reason": "Evaluation engine offline"}

    start_time = time.time()
    
    eval_input = {
        "final_narrative": narration,
        "causal_chains": chains,
        "raw_tools": tools
    }
    
    system_msg = STRICT_EVALUATOR_PROMPT
    user_msg = json.dumps(eval_input)

    trace = None
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(
            name="reasoning_evaluation",
            metadata={"stage": "audit"}
        )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Absolute determinism for auditing
        )
        
        latency = time.time() - start_time
        output_text = response.choices[0].message.content.strip()
        
        if trace:
            trace.generation(
                name="audit_call",
                input={"system": system_msg, "user": user_msg},
                output=output_text,
                model="llama-3.1-8b-instant",
                usage={
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None
                },
                metadata={"latency": latency}
            )
            langfuse.flush()
            
        return json.loads(output_text)
    except Exception as e:
        logger.error(f"Reasoning Audit Failed: {e}")
        return {"score": 0.0, "reason": f"Audit exception: {str(e)[:50]}"}
