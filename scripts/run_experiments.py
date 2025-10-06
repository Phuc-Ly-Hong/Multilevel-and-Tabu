import os
import subprocess
import sys
import re
from pathlib import Path
import time
import csv

def parse_output(output_text, instance_name, runtime):
    """Parse output từ C++ program để extract thông tin"""
    
    # Tìm số nodes
    nodes_match = re.search(r'Read (\d+) nodes', output_text)
    num_nodes = int(nodes_match.group(1)) if nodes_match else 0
    
    # Tìm segment length  
    segment_match = re.search(r'Segment length: (\d+)', output_text)
    segment_length = int(segment_match.group(1)) if segment_match else 0
    
    # Tìm kết quả cuối cùng
    lines = output_text.split('\n')
    
    # Tìm route details cuối cùng
    vehicle_routes = {}
    for i, line in enumerate(lines):
        if "Route details:" in line:
            # Đọc các routes tiếp theo
            j = i + 1
            while j < len(lines) and lines[j].startswith("Vehicle "):
                route_match = re.match(r'Vehicle (\d+): (.*)', lines[j].strip())
                if route_match:
                    vehicle_id = int(route_match.group(1))
                    route = route_match.group(2).strip()
                    vehicle_routes[f'Vehicle{vehicle_id}_Route'] = f'"{route}"'
                j += 1
    
    # Tìm fitness values cuối cùng (từ cuối lên)
    makespan = 0
    drone_violation = 0
    waiting_violation = 0
    fitness = 0
    is_feasible = True
    
    for line in reversed(lines):
        if "Makespan:" in line:
            makespan_match = re.search(r'Makespan: ([\d.]+)', line)
            if makespan_match:
                makespan = float(makespan_match.group(1))
        elif "Drone violation:" in line:
            drone_match = re.search(r'Drone violation: ([\d.]+)', line)
            if drone_match:
                drone_violation = float(drone_match.group(1))
        elif "Waiting violation:" in line:
            waiting_match = re.search(r'Waiting violation: ([\d.]+)', line)
            if waiting_match:
                waiting_violation = float(waiting_match.group(1))
        elif "Fitness:" in line:
            fitness_match = re.search(r'Fitness: ([\d.]+)', line)
            if fitness_match:
                fitness = float(fitness_match.group(1))
                break
    
    if drone_violation > 0 or waiting_violation > 0:
        is_feasible = False
    
    # Tìm iterations used
    iterations_used = 0
    iter_matches = re.findall(r'Iter: (\d+)', output_text)
    if iter_matches:
        iterations_used = int(iter_matches[-1]) + 1
    
    # Tạo result dict
    result = {
        'Instance': instance_name,
        'Makespan': makespan,
        'DroneViolation': drone_violation,
        'WaitingViolation': waiting_violation,
        'Fitness': fitness,
        'IsFeasible': is_feasible,
        'Iterations': iterations_used,
        'Runtime(s)': round(runtime, 2),
        'NumNodes': num_nodes,
        'SegmentLength': segment_length
    }
    
    # Thêm routes (tối đa 2 vehicles)
    for i in range(2):
        key = f'Vehicle{i}_Route'
        if key in vehicle_routes:
            result[key] = vehicle_routes[key]
        else:
            result[key] = '""'
    
    return result

def run_single_instance(cpp_file, instance_file, timeout=1800):
    """Chạy 1 instance và return kết quả"""
    
    instance_name = Path(instance_file).stem
    print(f"Running {instance_name}...")
    
    # Tạo file C++ tạm thời với đường dẫn instance
    temp_cpp = f"temp_{instance_name}.cpp"
    
    try:
        # Đọc file C++ gốc
        with open(cpp_file, 'r', encoding='utf-8') as f:
            cpp_content = f.read()
        
        # Thay thế đường dẫn dataset - SỬA CHO ĐÚNG CẤU TRÚC
        old_path_pattern = r'read_dataset\(".*?"\);'
        new_path = f'read_dataset("{instance_file}");'
        cpp_content = re.sub(old_path_pattern, new_path, cpp_content)
        
        # Ghi file tạm
        with open(temp_cpp, 'w', encoding='utf-8') as f:
            f.write(cpp_content)
        
        # Compile
        executable = f"temp_{instance_name}"
        if os.name == 'nt':  # Windows
            executable += ".exe"
            
        compile_cmd = ["g++", "-O2", "-std=c++17", "-o", executable, temp_cpp]
        print(f"  Compiling: {' '.join(compile_cmd)}")
        
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            print(f"  ✗ Compilation failed for {instance_name}")
            print(f"  Error: {compile_result.stderr}")
            return None
        
        # Run
        print(f"  Executing {executable}...")
        start_time = time.time()
        
        if os.name == 'nt':  # Windows
            run_result = subprocess.run([executable], capture_output=True, text=True, timeout=timeout)
        else:  # Linux/Mac
            run_result = subprocess.run([f"./{executable}"], capture_output=True, text=True, timeout=timeout)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if run_result.returncode != 0:
            print(f"  ✗ Execution failed for {instance_name}")
            print(f"  Error: {run_result.stderr}")
            return None
        
        # Parse kết quả
        result = parse_output(run_result.stdout, instance_name, runtime)
        
        print(f"  ✓ {instance_name} completed: Fitness={result['Fitness']:.2f}, Time={runtime:.1f}s")
        
        return result
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ {instance_name} timed out (>{timeout/60:.0f} min)")
        return None
    except Exception as e:
        print(f"  ✗ {instance_name} failed: {e}")
        return None
    finally:
        # Cleanup
        for file_to_remove in [temp_cpp, f"temp_{instance_name}", f"temp_{instance_name}.exe"]:
            if os.path.exists(file_to_remove):
                try:
                    os.remove(file_to_remove)
                except:
                    pass

def main():
    # Cấu hình
    cpp_file = "src/Multilevel_Tabu.cpp"  # SỬA: đường dẫn trong src/
    instances_dir = "instances"
    results_file = "results/results.csv"
    
    print("=== TABU SEARCH EXPERIMENT RUNNER ===")
    print(f"C++ source: {cpp_file}")
    print(f"Instances directory: {instances_dir}")
    print(f"Results file: {results_file}")
    
    # Tạo thư mục results nếu chưa có
    os.makedirs("results", exist_ok=True)
    
    # Kiểm tra files tồn tại
    if not os.path.exists(cpp_file):
        print(f"Error: {cpp_file} not found!")
        return
    
    if not os.path.exists(instances_dir):
        print(f"Error: {instances_dir} directory not found!")
        return
    
    # Lấy danh sách instances
    instance_files = []
    for f in os.listdir(instances_dir):
        if f.endswith('.txt'):
            instance_files.append(os.path.join(instances_dir, f))
    
    instance_files.sort()
    
    if not instance_files:
        print(f"No .txt files found in {instances_dir}")
        return
    
    print(f"\nFound {len(instance_files)} instances:")
    for i, f in enumerate(instance_files, 1):
        print(f"  {i}. {f}")
    
    # Chạy từng instance
    results = []
    successful = 0
    failed = 0
    total_time = 0
    
    start_time = time.time()
    
    for i, instance_file in enumerate(instance_files, 1):
        print(f"\n[{i}/{len(instance_files)}] Processing {instance_file}")
        print("-" * 50)
        
        result = run_single_instance(cpp_file, instance_file)
        
        if result:
            results.append(result)
            successful += 1
            total_time += result['Runtime(s)']
        else:
            failed += 1
    
    # Lưu kết quả
    if results:
        # Define column order
        columns = ['Instance', 'Makespan', 'DroneViolation', 'WaitingViolation', 
                  'Fitness', 'IsFeasible', 'Iterations', 'Runtime(s)', 'NumNodes', 
                  'SegmentLength', 'Vehicle0_Route', 'Vehicle1_Route']
        
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to {results_file}")
        
        # Summary
        end_time = time.time()
        total_experiment_time = end_time - start_time
        
        print(f"\n" + "=" * 60)
        print(f"EXPERIMENT SUMMARY")
        print(f"=" * 60)
        print(f"Total instances: {len(instance_files)}")
        print(f"Successful runs: {successful}")
        print(f"Failed runs: {failed}")
        print(f"Success rate: {successful/len(instance_files)*100:.1f}%")
        print(f"Total experiment time: {total_experiment_time/60:.1f} minutes")
        print(f"Average runtime per instance: {total_time/successful:.1f}s" if successful > 0 else "N/A")
        
        # Statistics
        if successful > 0:
            fitnesses = [r['Fitness'] for r in results]
            print(f"\nFITNESS STATISTICS:")
            print(f"Best fitness: {min(fitnesses):.2f}")
            print(f"Worst fitness: {max(fitnesses):.2f}")
            print(f"Average fitness: {sum(fitnesses)/len(fitnesses):.2f}")
        
        # Top 5 results
        if len(results) >= 5:
            print(f"\nTOP 5 RESULTS:")
            sorted_results = sorted(results, key=lambda x: x['Fitness'])
            for i, result in enumerate(sorted_results[:5], 1):
                feasible = "✓" if result['IsFeasible'] else "✗"
                print(f"{i}. {result['Instance']}: {result['Fitness']:.2f} {feasible} ({result['Runtime(s)']}s)")
    else:
        print("\n✗ No successful runs!")

if __name__ == "__main__":
    main()
