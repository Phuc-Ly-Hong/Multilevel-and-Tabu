import os
import subprocess
import sys
import re
from pathlib import Path
import time
import csv

def parse_output(output_text, instance_name, runtime):
    """Parse output từ C++ program để extract thông tin."""
    
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
        'VehicleRoutes': vehicle_routes
    }
    
    return result

def run_single_instance(cpp_file, instance_file, timeout=600):
    """Chạy 1 instance 1 lần và return kết quả"""
    
    instance_name = Path(instance_file).stem
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
        
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            print(f"    ✗ Compilation failed for {instance_name}")
            print(f"    Error: {compile_result.stderr}")
            return None
        
        # Run
        start_time = time.time()
        
        if os.name == 'nt':  # Windows
            run_result = subprocess.run([executable], capture_output=True, text=True, timeout=timeout)
        else:  # Linux/Mac
            run_result = subprocess.run([f"./{executable}"], capture_output=True, text=True, timeout=timeout)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if run_result.returncode != 0:
            print(f"    ✗ Execution failed for {instance_name}")
            print(f"    Error: {run_result.stderr}")
            return None
        
        # Parse kết quả
        result = parse_output(run_result.stdout, instance_name, runtime)
        
        return result
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ {instance_name} timed out (>{timeout/60:.0f} min)")
        return None
    except Exception as e:
        print(f"    ✗ {instance_name} failed: {e}")
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
    results_file = "results/single_run_results.csv"
    timeout = 600  # 10 phút timeout mỗi instance
    
    print("=== TABU SEARCH SINGLE RUN EXPERIMENT ===")
    print(f"C++ source: {cpp_file}")
    print(f"Instances directory: {instances_dir}")
    print(f"Timeout per instance: {timeout/60:.0f} minutes")
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
    all_results = []
    successful_instances = 0
    failed_instances = 0
    
    experiment_start_time = time.time()
    
    for i, instance_file in enumerate(instance_files, 1):
        print(f"\n[{i}/{len(instance_files)}] Processing {instance_file}")
        print("=" * 60)
        
        result = run_single_instance(cpp_file, instance_file, timeout)
        
        if result:
            all_results.append(result)
            successful_instances += 1
            
            print(f"    ✅ {result['Instance']}: Fitness={result['Fitness']:.2f}, "
                  f"Time={result['Runtime(s)']:.1f}s, Feasible={result['IsFeasible']}")
        else:
            failed_instances += 1
            print(f"    ❌ {Path(instance_file).stem} failed")
    
    # Lưu kết quả ra CSV
    if all_results:
        # Determine maximum number of vehicles across all runs
        max_vehicles = max(r['NumVehicles'] for r in all_results)
        
        # Define column order
        columns = ['Instance', 'Makespan', 'DroneViolation', 'WaitingViolation', 
                  'Fitness', 'IsFeasible', 'Iterations', 'Runtime(s)', 'NumNodes', 
                  'SegmentLength', 'NumVehicles']
        
        # Add vehicle route columns dynamically
        for v in range(max_vehicles):
            columns.append(f'Vehicle{v}_Route')
        
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for result in all_results:
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
        
        print(f"\n✅ Results saved to {results_file}")
    
    # Final experiment summary
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    print(f"\n" + "=" * 70)
    print(f"🎯 EXPERIMENT SUMMARY")
    print(f"=" * 70)
    print(f"📁 Total instances: {len(instance_files)}")
    print(f"✅ Successful: {successful_instances}")
    print(f"❌ Failed: {failed_instances}")
    print(f"📊 Success rate: {successful_instances/len(instance_files)*100:.1f}%")
    print(f"⏱️  Total time: {total_experiment_time/60:.1f} minutes")
    print(f"⏱️  Average per instance: {total_experiment_time/len(instance_files):.1f} seconds")
    
    if all_results:
        # Thống kê cơ bản
        fitnesses = [r['Fitness'] for r in all_results]
        runtimes = [r['Runtime(s)'] for r in all_results]
        feasible_count = sum(1 for r in all_results if r['IsFeasible'])
        
        print(f"\n📈 PERFORMANCE STATISTICS:")
        print(f"Feasible solutions: {feasible_count}/{len(all_results)} ({feasible_count/len(all_results)*100:.1f}%)")
        print(f"Best fitness: {min(fitnesses):.2f}")
        print(f"Worst fitness: {max(fitnesses):.2f}")
        print(f"Average fitness: {sum(fitnesses)/len(fitnesses):.2f}")
        print(f"Average runtime: {sum(runtimes)/len(runtimes):.1f} seconds")
        
        # Top 10 best instances
        print(f"\n🏆 TOP 10 BEST INSTANCES:")
        sorted_results = sorted(all_results, key=lambda x: x['Fitness'])
        for i, result in enumerate(sorted_results[:10], 1):
            feasible_str = "✅" if result['IsFeasible'] else "❌"
            print(f"{i:2d}. {result['Instance']:15s}: {result['Fitness']:8.2f} "
                  f"({result['Runtime(s)']:5.1f}s) {feasible_str}")
    
    print(f"\n📄 Results saved to: {results_file}")
    print(f"🎉 Experiment completed!")

if __name__ == "__main__":
    main()