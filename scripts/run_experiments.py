import os
import subprocess
import sys
import re
from pathlib import Path
import time
import csv

def parse_output(output_text, instance_name, runtime):
    """Parse output từ C++ program để extract thông tin (hỗ trợ nhiều vehicle động)."""
    
    # Tìm số nodes
    nodes_match = re.search(r'Read (\d+) nodes', output_text)
    num_nodes = int(nodes_match.group(1)) if nodes_match else 0
    
    # Tìm segment length  
    segment_match = re.search(r'Segment length: (\d+)', output_text)
    segment_length = int(segment_match.group(1)) if segment_match else 0
    
    lines = output_text.splitlines()
    
    # Tìm section "Route details:" cuối cùng và đọc tất cả vehicles
    vehicle_routes = {}
    last_route_section = -1
    
    # Tìm section "Route details:" cuối cùng
    for i in range(len(lines) - 1, -1, -1):
        if "Route details:" in lines[i]:
            last_route_section = i
            break
    
    if last_route_section != -1:
        # Đọc các routes từ section cuối cùng
        j = last_route_section + 1
        while j < len(lines):
            line = lines[j].strip()
            if not line.startswith("Vehicle "):
                break
            route_match = re.match(r'Vehicle\s+(\d+)\s*:\s*(.*)', line)
            if route_match:
                vehicle_id = int(route_match.group(1))
                route = route_match.group(2).strip()
                vehicle_routes[vehicle_id] = route
            j += 1
    
    # Tìm fitness values cuối cùng (từ cuối lên)
    makespan = 0.0
    drone_violation = 0.0
    waiting_violation = 0.0
    fitness = 0.0
    is_feasible = True
    
    for line in reversed(lines):
        if "Makespan:" in line:
            makespan_match = re.search(r'Makespan:\s*([\d.]+)', line)
            if makespan_match:
                makespan = float(makespan_match.group(1))
        elif "Drone violation:" in line:
            drone_match = re.search(r'Drone violation:\s*([\d.]+)', line)
            if drone_match:
                drone_violation = float(drone_match.group(1))
        elif "Waiting violation:" in line:
            waiting_match = re.search(r'Waiting violation:\s*([\d.]+)', line)
            if waiting_match:
                waiting_violation = float(waiting_match.group(1))
        elif "Fitness:" in line:
            fitness_match = re.search(r'Fitness:\s*([\d.]+)', line)
            if fitness_match:
                fitness = float(fitness_match.group(1))
                break
    
    if drone_violation > 0 or waiting_violation > 0:
        is_feasible = False
    
    # Tìm iterations used
    iterations_used = 0
    iter_matches = re.findall(r'Iter:\s*(\d+)', output_text)
    if iter_matches:
        iterations_used = int(iter_matches[-1]) + 1
    
    # Đếm số vehicles
    num_vehicles = len(vehicle_routes)
    
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
        'SegmentLength': segment_length,
        'NumVehicles': num_vehicles,
        'VehicleRoutes': vehicle_routes  # dict: vehicle_id -> route_string
    }
    
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
        
        # Thay thế đường dẫn dataset
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
        
        print(f"  ✓ {instance_name} completed: Fitness={result['Fitness']:.2f}, Time={runtime:.1f}s, Vehicles={result['NumVehicles']}")
        
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
    cpp_file = "src/Multilevel_Tabu.cpp"
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
        # Determine maximum number of vehicles across all runs
        max_vehicles = max(r['NumVehicles'] for r in results)
        
        # Define column order - dynamic vehicle columns
        columns = ['Instance', 'Makespan', 'DroneViolation', 'WaitingViolation', 
                  'Fitness', 'IsFeasible', 'Iterations', 'Runtime(s)', 'NumNodes', 
                  'SegmentLength', 'NumVehicles']
        
        # Add vehicle route columns dynamically
        for v in range(max_vehicles):
            columns.append(f'Vehicle{v}_Route')
        
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for result in results:
                # Prepare row data
                row = {}
                for col in columns:
                    if col.startswith('Vehicle') and col.endswith('_Route'):
                        # Extract vehicle number
                        vehicle_num = int(col.replace('Vehicle', '').replace('_Route', ''))
                        vehicle_routes = result.get('VehicleRoutes', {})
                        route_str = vehicle_routes.get(vehicle_num, "")
                        row[col] = f'"{route_str}"' if route_str else '""'
                    else:
                        row[col] = result.get(col, "")
                
                writer.writerow(row)
        
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
        print(f"Max vehicles used: {max_vehicles}")
        
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
                print(f"{i}. {result['Instance']}: {result['Fitness']:.2f} {feasible} ({result['Runtime(s)']}s, {result['NumVehicles']} vehicles)")
    else:
        print("\n✗ No successful runs!")

if __name__ == "__main__":
    main()
