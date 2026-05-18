import subprocess
import sys
import os

# Check if we're being run by Streamlit Cloud
if "streamlit" in sys.argv[0] or os.environ.get("STREAMLIT_SERVER_PORT"):
    print("🚀 Streamlit runtime detected. Executing dashboard directly...")
    # Execute the actual UI script in the current Streamlit context
    with open("app/ui/dashboard.py", encoding="utf-8") as f:
        code = compile(f.read(), "app/ui/dashboard.py", "exec")
        exec(code, globals())
else:
    def main():
        print("🚀 Launching AlphaTrace Intelligence Dashboard...")
        try:
            # POINT DIRECTLY TO THE UI ENTRYPOINT
            cmd = ["streamlit", "run", "app/ui/dashboard.py", "--server.runOnSave", "false"]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped.")
        except Exception as e:
            print(f"❌ Failed to launch dashboard: {e}")

    if __name__ == "__main__":
        main()
