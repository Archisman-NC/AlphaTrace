import subprocess
import sys

try:
    import streamlit.runtime
    in_streamlit = streamlit.runtime.exists()
except Exception:
    in_streamlit = False

if in_streamlit:
    print("🚀 Streamlit runtime detected. Executing dashboard directly...")
    with open("app/ui/dashboard.py", encoding="utf-8") as f:
        code = compile(f.read(), "app/ui/dashboard.py", "exec")
        exec(code, globals())
else:
    def main():
        print("🚀 Launching AlphaTrace Intelligence Dashboard...")
        try:
            cmd = ["streamlit", "run", "app/ui/dashboard.py", "--server.runOnSave", "false"]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped.")
        except Exception as e:
            print(f"❌ Failed to launch dashboard: {e}")

    if __name__ == "__main__":
        main()
