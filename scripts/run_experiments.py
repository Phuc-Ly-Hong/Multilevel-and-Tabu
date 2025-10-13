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

def run_single_instance(cpp_file, instance_file, timeout=1800, run_id=1):
    """Chạy 1 instance và return kết quả (với run_id để tạo file unique)"""
    
    instance_name = Path(instance_file).stem
    
    # Tạo file C++ tạm thời với đường dẫn instance (unique cho mỗi run)
    temp_cpp = f"temp_{instance_name}_run{run_id}.cpp"
    
    try:
        # Đọc file C++ gốc
        with open(cpp_file, 'r', encoding='utf-8') as f:
            cpp_content = f.read()
        
        # Thay thế đường dẫn dataset
        old_path_pattern = r'read_dataset\(".*?"\);'
        new_path = f'read_dataset("{instance_file}");'
        cpp_content = re.sub(old_path_pattern, new_path, cpp_content)
        
        # Thay đổi random seed cho mỗi run để có kết quả khác nhau
        seed_replacement = f'srand(time(nullptr) + {run_id * 1000});'
        cpp_content = re.sub(r'srand\(time\(nullptr\)\);', seed_replacement, cpp_content)
        
        # Ghi file tạm
        with open(temp_cpp, 'w', encoding='utf-8') as f:
            f.write(cpp_content)
        
        # Compile
        executable = f"temp_{instance_name}_run{run_id}"
        if os.name == 'nt':  # Windows
            executable += ".exe"
            
        compile_cmd = ["g++", "-O2", "-std=c++17", "-o", executable, temp_cpp]
        
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            print(f"    ✗ Compilation failed for {instance_name} run {run_id}")
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
            print(f"    ✗ Execution failed for {instance_name} run {run_id}")
            print(f"    Error: {run_result.stderr}")
            return None
        
        # Parse kết quả
        result = parse_output(run_result.stdout, instance_name, runtime)
        
        return result
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ {instance_name} run {run_id} timed out (>{timeout/60:.0f} min)")
        return None
    except Exception as e:
        print(f"    ✗ {instance_name} run {run_id} failed: {e}")
        return None
    finally:
        # Cleanup
        for file_to_remove in [temp_cpp, f"temp_{instance_name}_run{run_id}", f"temp_{instance_name}_run{run_id}.exe"]:
            if os.path.exists(file_to_remove):
                try:
                    os.remove(file_to_remove)
                except:
                    pass

def run_multiple_runs(cpp_file, instance_file, num_runs=5, timeout=1800):
    """Chạy 1 instance nhiều lần và return list kết quả"""
    
    instance_name = Path(instance_file).stem
    print(f"Running {instance_name} for {num_runs} runs...")
    
    results = []
    
    for run in range(1, num_runs + 1):
        print(f"  Run {run}/{num_runs}...")
        
        result = run_single_instance(cpp_file, instance_file, timeout, run_id=run)
        if result:
            result['Run'] = run
            results.append(result)
            print(f"    ✓ Run {run}: Fitness={result['Fitness']:.2f}, Time={result['Runtime(s)']:.1f}s, Feasible={result['IsFeasible']}")
        else:
            print(f"    ✗ Run {run} failed")
    
    return results

def calculate_statistics(values):
    """Tính toán thống kê cơ bản"""
    if not values:
        return {'min': 0, 'max': 0, 'avg': 0, 'std': 0}
    
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)
    
    variance = sum((x - avg_val) ** 2 for x in values) / len(values)
    std_val = variance ** 0.5
    
    return {
        'min': min_val,
        'max': max_val,
        'avg': avg_val,
        'std': std_val
    }

def main():
    # Cấu hình
    cpp_file = "src/Multilevel_Tabu.cpp"
    instances_dir = "instances"
    detailed_results_file = "results/detailed_results.csv"
    summary_results_file = "results/summary_results.csv"
    num_runs = 5  # Số lần chạy mỗi instance
    timeout = 1800  # 30 phút timeout
    
    print("=== TABU SEARCH MULTI-RUN EXPERIMENT ===")
    print(f"C++ source: {cpp_file}")
    print(f"Instances directory: {instances_dir}")
    print(f"Runs per instance: {num_runs}")
    print(f"Timeout per run: {timeout/60:.0f} minutes")
    print(f"Detailed results: {detailed_results_file}")
    print(f"Summary results: {summary_results_file}")
    
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
    
    # Chạy từng instance multiple runs
    all_detailed_results = []
    summary_results = []
    successful_instances = 0
    failed_instances = 0
    total_runs = 0
    successful_runs = 0
    
    experiment_start_time = time.time()
    
    for i, instance_file in enumerate(instance_files, 1):
        print(f"\n[{i}/{len(instance_files)}] Processing {instance_file}")
        print("=" * 70)
        
        instance_results = run_multiple_runs(cpp_file, instance_file, num_runs, timeout)
        
        if instance_results:
            # Thêm vào detailed results
            all_detailed_results.extend(instance_results)
            successful_instances += 1
            successful_runs += len(instance_results)
            
            # Tính thống kê cho instance này
            fitnesses = [r['Fitness'] for r in instance_results]
            runtimes = [r['Runtime(s)'] for r in instance_results]
            feasible_count = sum(1 for r in instance_results if r['IsFeasible'])
            
            fitness_stats = calculate_statistics(fitnesses)
            runtime_stats = calculate_statistics(runtimes)
            
            instance_name = Path(instance_file).stem
            summary = {
                'Instance': instance_name,
                'TotalRuns': len(instance_results),
                'SuccessfulRuns': len(instance_results),
                'FeasibleRuns': feasible_count,
                'FeasibilityRate(%)': round(feasible_count / len(instance_results) * 100, 1),
                'BestFitness': round(fitness_stats['min'], 2),
                'WorstFitness': round(fitness_stats['max'], 2),
                'AvgFitness': round(fitness_stats['avg'], 2),
                'StdDevFitness': round(fitness_stats['std'], 2),
                'MinRuntime(s)': round(runtime_stats['min'], 2),
                'MaxRuntime(s)': round(runtime_stats['max'], 2),
                'AvgRuntime(s)': round(runtime_stats['avg'], 2),
                'TotalRuntime(s)': round(sum(runtimes), 2)
            }
            summary_results.append(summary)
            
            print(f"\n  📊 {instance_name} SUMMARY:")
            print(f"    Completed runs: {len(instance_results)}/{num_runs}")
            print(f"    Feasible runs: {feasible_count}/{num_runs} ({feasible_count/num_runs*100:.1f}%)")
            print(f"    Best fitness: {fitness_stats['min']:.2f}")
            print(f"    Avg fitness: {fitness_stats['avg']:.2f} ± {fitness_stats['std']:.2f}")
            print(f"    Avg runtime: {runtime_stats['avg']:.1f}s")
            print(f"    Total time: {sum(runtimes)/60:.1f} minutes")
        else:
            failed_instances += 1
            print(f"  ❌ {Path(instance_file).stem} failed completely")
        
        total_runs += num_runs
    
    # Lưu detailed results
    if all_detailed_results:
        # Determine maximum number of vehicles across all runs
        max_vehicles = max(r['NumVehicles'] for r in all_detailed_results)
        
        # Define column order - dynamic vehicle columns
        columns = ['Instance', 'Run', 'Makespan', 'DroneViolation', 'WaitingViolation', 
                  'Fitness', 'IsFeasible', 'Iterations', 'Runtime(s)', 'NumNodes', 
                  'SegmentLength', 'NumVehicles']
        
        # Add vehicle route columns dynamically
        for v in range(max_vehicles):
            columns.append(f'Vehicle{v}_Route')
        
        with open(detailed_results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for result in all_detailed_results:
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
        
        print(f"\n✅ Detailed results saved to {detailed_results_file}")
    
    # Lưu summary results
    if summary_results:
        summary_columns = ['Instance', 'TotalRuns', 'SuccessfulRuns', 'FeasibleRuns', 'FeasibilityRate(%)',
                          'BestFitness', 'WorstFitness', 'AvgFitness', 'StdDevFitness', 
                          'MinRuntime(s)', 'MaxRuntime(s)', 'AvgRuntime(s)', 'TotalRuntime(s)']
        
        with open(summary_results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_columns)
            writer.writeheader()
            writer.writerows(summary_results)
        
        print(f"✅ Summary results saved to {summary_results_file}")
    
    # Final experiment summary
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    print(f"\n" + "=" * 80)
    print(f"🎯 FINAL EXPERIMENT SUMMARY")
    print(f"=" * 80)
    print(f"📁 Total instances processed: {len(instance_files)}")
    print(f"✅ Successful instances: {successful_instances}")
    print(f"❌ Failed instances: {failed_instances}")
    print(f"🏃 Total runs attempted: {total_runs}")
    print(f"✅ Successful runs: {successful_runs}")
    print(f"📊 Success rate: {successful_runs/total_runs*100:.1f}%")
    print(f"⏱️  Total experiment time: {total_experiment_time/60:.1f} minutes")
    
    if summary_results:
        # Tính thống kê tổng quát
        all_best_fitnesses = [s['BestFitness'] for s in summary_results]
        all_avg_fitnesses = [s['AvgFitness'] for s in summary_results]
        all_feasibility_rates = [s['FeasibilityRate(%)'] for s in summary_results]
        
        overall_stats = calculate_statistics(all_best_fitnesses)
        
        print(f"\n🏆 OVERALL PERFORMANCE:")
        print(f"Best fitness found: {min(all_best_fitnesses):.2f}")
        print(f"Average best fitness: {sum(all_best_fitnesses)/len(all_best_fitnesses):.2f}")
        print(f"Average feasibility rate: {sum(all_feasibility_rates)/len(all_feasibility_rates):.1f}%")
        
        # Top 5 instances by best fitness
        print(f"\n🥇 TOP 5 INSTANCES (by best fitness):")
        sorted_summary = sorted(summary_results, key=lambda x: x['BestFitness'])
        for i, result in enumerate(sorted_summary[:5], 1):
            feasible_rate = result['FeasibilityRate(%)']
            print(f"{i}. {result['Instance']}: {result['BestFitness']:.2f} "
                  f"(avg: {result['AvgFitness']:.2f}, feasible: {feasible_rate}%)")
    
    print(f"\n📄 Check results files:")
    print(f"  - Detailed: {detailed_results_file}")
    print(f"  - Summary: {summary_results_file}")
    print(f"\n🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()