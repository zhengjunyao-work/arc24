#!/usr/bin/env python3
"""
Test GPU performance and provide optimization recommendations for Mac Mini
"""

import torch
import torch.nn as nn
import time
import numpy as np
from VAEModel import VAE1D

def test_device_performance():
    """Test performance of different devices"""
    print("GPU Performance Test for Mac Mini")
    print("=" * 50)
    
    # Test parameters
    batch_sizes = [16, 32, 64, 128]
    input_length = 1124
    latent_dim = 64
    hidden_dims = [512, 256, 128]
    num_heads = 8
    
    # Check available devices
    devices = []
    if torch.backends.mps.is_available():
        devices.append(("MPS (Apple Silicon)", torch.device("mps")))
    if torch.cuda.is_available():
        devices.append(("CUDA (NVIDIA)", torch.device("cuda")))
    devices.append(("CPU", torch.device("cpu")))
    
    print(f"Available devices: {[d[0] for d in devices]}")
    
    results = {}
    
    for device_name, device in devices:
        print(f"\nTesting {device_name}...")
        print("-" * 30)
        
        device_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create model
            model = VAE1D(
                input_length=input_length,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                num_heads=num_heads,
                use_input_norm=True,
                use_batch_norm=True
            ).to(device)
            
            # Create test data
            test_data = torch.randn(batch_size, input_length).to(device)
            
            # Warm up
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = model(test_data)
            
            # Test inference speed
            model.eval()
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.time()
                    _ = model(test_data)
                    if device.type != 'cpu':
                        torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            device_results[batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput
            }
            
            print(f"    Avg time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"    Throughput: {throughput:.2f} samples/sec")
        
        results[device_name] = device_results
    
    return results

def test_memory_usage():
    """Test memory usage on different devices"""
    print(f"\nMemory Usage Test")
    print("=" * 30)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Testing MPS memory usage...")
        
        # Test different model sizes
        model_sizes = [
            ("Small", [256, 128, 64]),
            ("Medium", [512, 256, 128]),
            ("Large", [1024, 512, 256])
        ]
        
        for size_name, hidden_dims in model_sizes:
            try:
                model = VAE1D(
                    input_length=1124,
                    latent_dim=64,
                    hidden_dims=hidden_dims,
                    num_heads=8,
                    use_input_norm=True,
                    use_batch_norm=True
                ).to(device)
                
                # Test with different batch sizes
                for batch_size in [16, 32, 64]:
                    try:
                        test_data = torch.randn(batch_size, 1124).to(device)
                        _ = model(test_data)
                        print(f"  {size_name} model, batch_size={batch_size}: ✅ Success")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"  {size_name} model, batch_size={batch_size}: ❌ OOM")
                        else:
                            print(f"  {size_name} model, batch_size={batch_size}: ❌ Error: {e}")
                
                del model
                torch.mps.empty_cache()
                
            except Exception as e:
                print(f"  {size_name} model: ❌ Error: {e}")

def provide_recommendations(results):
    """Provide optimization recommendations based on test results"""
    print(f"\nOptimization Recommendations")
    print("=" * 40)
    
    if not results:
        print("No test results available")
        return
    
    # Find best performing device
    best_device = None
    best_throughput = 0
    
    for device_name, device_results in results.items():
        for batch_size, metrics in device_results.items():
            if metrics['throughput'] > best_throughput:
                best_throughput = metrics['throughput']
                best_device = (device_name, batch_size)
    
    if best_device:
        print(f"✅ Best performance: {best_device[0]} with batch_size={best_device[1]}")
        print(f"   Throughput: {best_throughput:.2f} samples/sec")
    
    # MPS-specific recommendations
    if torch.backends.mps.is_available():
        print(f"\n🍎 Apple Silicon (MPS) Optimizations:")
        print(f"  • Use batch_size=64 or 128 for optimal GPU utilization")
        print(f"  • Enable memory efficient attention if available")
        print(f"  • Set memory fraction to 0.8 to avoid OOM errors")
        print(f"  • Use pin_memory=True in DataLoader")
        print(f"  • Consider using persistent_workers=True")
    
    # General recommendations
    print(f"\n🚀 General Performance Tips:")
    print(f"  • Increase batch_size until memory is full")
    print(f"  • Use mixed precision training (automatic on MPS)")
    print(f"  • Enable gradient clipping to prevent instability")
    print(f"  • Use learning rate scheduling")
    print(f"  • Monitor memory usage and adjust batch_size accordingly")

def main():
    """Main function to run performance tests"""
    print("Mac Mini GPU Performance Analysis")
    print("=" * 50)
    
    # Test device performance
    results = test_device_performance()
    
    # Test memory usage
    test_memory_usage()
    
    # Provide recommendations
    provide_recommendations(results)
    
    print(f"\n✅ Performance analysis complete!")
    print(f"Use these results to optimize your training configuration.")

if __name__ == "__main__":
    main()
