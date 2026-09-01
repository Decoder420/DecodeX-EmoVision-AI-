import time
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import build_cnn_model, build_mobilenet_model

def benchmark_model(model, name, num_warmup=10, num_runs=100):
    dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = model(dummy_input, training=False)

    # Latency measurement
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    return {
        "Name": name,
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "Avg Latency (ms)": avg_latency,
        "P95 Latency (ms)": p95_latency,
        "Throughput (FPS)": fps
    }

def run_benchmark():
    print("=" * 75)
    print("           Facial Emotion Detection Architecture Benchmark")
    print("=" * 75)

    cnn = build_cnn_model()
    mobilenet = build_mobilenet_model(pretrained=False)

    results = [
        benchmark_model(cnn, "4-Block Custom CNN"),
        benchmark_model(mobilenet, "MobileNetV3-Small (Transfer)")
    ]

    print(f"{'Model Architecture':<30} | {'Total Params':<13} | {'Avg Latency':<12} | {'FPS':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['Name']:<30} | {r['Total Params']:<13,d} | {r['Avg Latency (ms)']:<6.2f} ms   | {r['Throughput (FPS)']:<6.1f}")
    print("=" * 75)
    print("\n* Note on Trade-offs:")
    print("  - 4-Block CNN is specifically tailored for 48x48 single-channel grayscale, offering fast inference and minimal memory.")
    print("  - MobileNetV3-Small adds feature extraction depth and receptive field through 1x1 conv projection and 2x upsampling,")
    print("    at the cost of additional parameter count and slightly higher latency.\n")

if __name__ == '__main__':
    run_benchmark()
