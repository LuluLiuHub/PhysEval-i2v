#!/usr/bin/env python3
"""
Simple Parallel VBench Evaluation
Generate videos with self-forcing, evaluate with VBench (no guidance comparison)
"""

import subprocess
import threading
import time
import sys

def run_gpu_script(script_name, gpu_id, results_container):
    """Run a GPU script and capture results"""
    print(f"🚀 Starting {script_name} on GPU {gpu_id}")

    try:
        # Pass through all command line arguments
        result = subprocess.run([
            "python", script_name
        ] + sys.argv[1:], capture_output=True, text=True, check=True)

        results_container[script_name] = {
            "success": True,
            "gpu_id": gpu_id,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        print(f"✅ {script_name} (GPU {gpu_id}) completed successfully")

    except subprocess.CalledProcessError as e:
        results_container[script_name] = {
            "success": False,
            "gpu_id": gpu_id,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode
        }
        print(f"❌ {script_name} (GPU {gpu_id}) failed with return code {e.returncode}")

def main():
    print("🚀 Simple Parallel VBench Evaluation")
    print("=" * 60)
    print("• Self-forcing video generation")
    print("• GPU 0: Temporal dimensions")
    print("• GPU 1: Non-temporal dimensions")
    print("• VBench evaluation (no guidance comparison)")
    print("=" * 60)

    # Prepare result containers
    results = {}

    # Create threads for both GPUs
    gpu0_thread = threading.Thread(
        target=run_gpu_script,
        args=("vbench_gpu0_temporal_simple.py", 0, results)
    )

    gpu1_thread = threading.Thread(
        target=run_gpu_script,
        args=("vbench_gpu1_nontemporal_simple.py", 1, results)
    )

    # Start both threads
    start_time = time.time()

    print("🏃‍♂️ Starting parallel execution...")
    gpu0_thread.start()
    time.sleep(2)  # Stagger start slightly
    gpu1_thread.start()

    # Wait for both to complete
    gpu0_thread.join()
    gpu1_thread.join()

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)

    # Show results
    for script, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{script}: {status} (GPU {result['gpu_id']})")
        if not result["success"] and "stderr" in result:
            print(f"   Error: {result['stderr'][:100]}...")

    print(f"\n⏱️ Total execution time: {total_time/60:.1f} minutes")

    if all(r["success"] for r in results.values()):
        print("🎉 Parallel evaluation completed successfully!")
    else:
        print("⚠️ Some evaluations failed. Check individual results.")

    print("\n📁 Result locations:")
    print("  • GPU 0 (temporal): ./eval_results_gpu0_temporal/")
    print("  • GPU 1 (non-temporal): ./eval_results_gpu1_nontemporal/")

if __name__ == "__main__":
    main()