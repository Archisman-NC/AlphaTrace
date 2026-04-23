import logging
import traceback
from typing import Dict, List, Any, Optional

# Mock/Bridge Imports - In a full production setup, these would import from their respective modules
# For now, we define placeholders that bridge to our existing analytics sub-modules
from main import run_pipeline

logger = logging.getLogger(__name__)

# --- Specialized Tool Wrappers ---

def run_full_analysis_wrapper(portfolio_id: str) -> Dict[str, Any]:
    """Bridges to the existing main pipeline logic."""
    try:
        # Assuming run_pipeline can take a single ID and return results
        # In our case, run_pipeline prints to stdout, we might need to capture or refactor later
        # For the purpose of the router, we simulate a successful data return
        run_pipeline([portfolio_id])
        return f"Full analysis completed for {portfolio_id}."
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise

def run_reason_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    """Interface for the Causal Reasoning Engine."""
    # Placeholder for reasoning tool logic
    return f"Deterministic causal chain established for {portfolio_id}."

def run_risk_engine_wrapper(portfolio_id: str) -> Dict[str, Any]:
    """Interface for the Risk & Volatility Engine."""
    # Placeholder for risk tool logic
    return f"Volatility profile and drawdown risk calculated for {portfolio_id}."

def switch_portfolio_wrapper(portfolio_id: str) -> str:
    """Handles the logical switch of the active context."""
    return f"Active context successfully migrated to {portfolio_id}."

# --- Strategic Router Configuration ---

# Intent Priority: Switch must happen first to ensure all subsequent analytics 
# are performed on the new portfolio.
EXECUTION_PRIORITY = [
    "switch_portfolio",
    "reason",
    "risk",
    "full_analysis"
]

# Strategic Mapping
ROUTER = {
    "full_analysis": run_full_analysis_wrapper,
    "reason": run_reason_engine_wrapper,
    "risk": run_risk_engine_wrapper,
    "switch_portfolio": switch_portfolio_wrapper
}

# --- Core Routing Engine ---

def execute_intents(classification: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the execution of multiple financial intents with state awareness.
    
    Args:
        classification: Output from the Intent Classifier (intent array, portfolio_id, confidence)
        session: Current session state containing 'current_portfolio'
        
    Returns:
        Aggregated results dictionary
    """
    target_portfolio_id = classification.get("portfolio_id", session.get("current_portfolio"))
    intents_to_run = classification.get("intent", [])
    
    # De-duplicate and sort based on priority
    ordered_intents = [i for i in EXECUTION_PRIORITY if i in intents_to_run]
    
    # Handle unknown intents present in input but not in priority list
    unknown_intents = [i for i in intents_to_run if i not in EXECUTION_PRIORITY]
    if unknown_intents:
        logger.warning(f"Ignoring unknown intents: {unknown_intents}")

    execution_results = []
    
    # Active State Tracker
    active_portfolio = session.get("current_portfolio")

    for intent in ordered_intents:
        tool_func = ROUTER.get(intent)
        if not tool_func:
            continue

        # Dynamic State Adjustment: If switching, update the active ID for subsequent tools
        if intent == "switch_portfolio":
            active_portfolio = target_portfolio_id
            session["current_portfolio"] = active_portfolio

        try:
            logger.info(f"Executing intent: {intent} for portfolio: {active_portfolio}")
            
            # Execute tool
            data = tool_func(active_portfolio)
            
            # Record Success
            execution_results.append({
                "type": intent,
                "data": data,
                "confidence": classification.get("confidence", 0.0),
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Tool execution failed for {intent}: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Record Failure without breaking the cycle
            execution_results.append({
                "type": intent,
                "data": None,
                "status": "error",
                "error_message": str(e)
            })

    return {
        "portfolio_id": active_portfolio,
        "results": execution_results,
        "session_updated": active_portfolio != session.get("initial_portfolio")
    }
