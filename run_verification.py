import subprocess
import os
import sys
import time

VERIFICATION_SCRIPTS = [
    "scratch/verify_market_data.py",
    "scratch/verify_signals.py",
    "scratch/verify_backtester.py",
    "scratch/verify_regime_detector.py",
    "scratch/verify_watchdog.py",
    "scratch/verify_signal_generator.py",
    "scratch/verify_portfolio_signals.py",
    "scratch/verify_rl_env.py",
    "scratch/verify_ppo_pipeline.py",
    "scratch/verify_research_evaluator.py",
    "scratch/verify_reasoning_integration.py"
]

def main():
    print("🛠️  AlphaTrace Unified Verification Suite")
    print("-" * 40)
    
    results = []
    start_time = time.time()
    
    for script in VERIFICATION_SCRIPTS:
        print(f"Running {script}...", end=" ", flush=True)
        try:
            # We run in a clean process to avoid state contamination
            process = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                print("✅ PASS")
                results.append((script, "PASS", ""))
            else:
                print("❌ FAIL")
                results.append((script, "FAIL", process.stderr or process.stdout))
        except Exception as e:
            print("❌ ERROR")
            results.append((script, "ERROR", str(e)))

    end_time = time.time()
    print("-" * 40)
    print(f"🏁 Verification Complete in {end_time - start_time:.2f}s")
    
    pass_count = sum(1 for _, status, _ in results if status == "PASS")
    print(f"📈 Result: {pass_count}/{len(VERIFICATION_SCRIPTS)} Passed")
    
    if pass_count < len(VERIFICATION_SCRIPTS):
        print("\n⚠️  Failures detected in:")
        for script, status, error in results:
            if status != "PASS":
                print(f"  - {script}")
        sys.exit(1)
    else:
        print("\n✨ All systems operational.")
        sys.exit(0)

if __name__ == "__main__":
    main()
