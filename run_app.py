import os
import sys
import subprocess

if __name__ == "__main__":
    # This is a legacy entrypoint. Forwarding to the actual UI script via streamlit
    print("🔔 Note: 'run_app.py' is deprecated. Please use 'run_dashboard.py' in the future.")
    try:
        # Point to the actual UI entrypoint, NOT back to run_dashboard.py
        cmd = ["streamlit", "run", "app/ui/dashboard.py", "--server.runOnSave", "false"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
