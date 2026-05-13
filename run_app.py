import os
import subprocess
import sys

def main():
    dashboard_path = os.path.join("app", "ui", "dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    print("🚀 Launching AlphaTrace Dashboard...")
    try:
        subprocess.run(["streamlit", "run", dashboard_path])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")

if __name__ == "__main__":
    main()
