import subprocess
import os
import sys

def main():
    print("🚀 Launching AlphaTrace Intelligence Dashboard...")
    try:
        # We call the existing run_app.py or streamlit directly
        cmd = ["streamlit", "run", "run_app.py", "--server.runOnSave", "false"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")

if __name__ == "__main__":
    main()
