import subprocess
import os
import sys

def main():
    print("🧪 Running AlphaTrace Comparative RL Research Pipeline...")
    try:
        # Run the comparative evaluation script
        cmd = [sys.executable, "research/evaluate_rl_vs_signals.py"]
        subprocess.run(cmd)
        
        print("\n✅ Research execution complete.")
        print("📁 Artifacts saved in: research/artifacts/")
    except Exception as e:
        print(f"❌ Research pipeline failed: {e}")

if __name__ == "__main__":
    main()
