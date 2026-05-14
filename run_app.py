import os
import sys
import subprocess

if __name__ == "__main__":
    # This is a legacy entrypoint. Forwarding to run_dashboard.py
    print("🔔 Note: 'run_app.py' is deprecated. Please use 'run_dashboard.py' in the future.")
    try:
        cmd = [sys.executable, "run_dashboard.py"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
