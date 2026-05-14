import subprocess
import os
import sys

def main():
    print("🚀 Launching AlphaTrace Intelligence Dashboard...")
    try:
        # POINT DIRECTLY TO THE UI ENTRYPOINT
        # This avoids the circular dependency with run_app.py
        cmd = ["streamlit", "run", "app/ui/dashboard.py", "--server.runOnSave", "false"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")

if __name__ == "__main__":
    main()
