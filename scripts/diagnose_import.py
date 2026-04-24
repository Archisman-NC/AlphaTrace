import sys
import os

# Add root to path to ensure absolute imports work
sys.path.append(os.getcwd())

try:
    import app.evaluation.llm_evaluator as mod
    print(f"MODULE_FILE: {mod.__file__}")
    print(f"MODULE_ATTRS: {dir(mod)}")
    if hasattr(mod, 'evaluate_response'):
        print("SUCCESS: evaluate_response found in module.")
    else:
        print("FAILURE: evaluate_response NOT found in module.")
except Exception as e:
    print(f"IMPORT_CRASH: {e}")
