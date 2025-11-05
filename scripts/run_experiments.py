import os
import subprocess
import sys
import re
from pathlib import Path
import time
import csv
from datetime import datetime

def parse_output(output_text, instance_name, runtime):
    """Parse output từ C++ program để extract thông tin."""
    
    # Tìm số nodes
    nodes_match = re.search(r'Read (\d+) nodes', output_text)
    num_nodes = int(nodes_match.group(1)) if nodes_match else 0
    
    # Tìm segment length  
    segment_match = re.search(r'Segment length: (\d+)', output_text)
    segment_length = int(segment_match.group(1)) if segment_match else 0
    
    lines = output_text.splitlines()

    # Parse thông tin từng level trong coarsening phase
    level_info = []
    level_pattern = re.compile(r'--- LEVEL (\d+) ---')
    fitness_pattern = re.compile(r'Fitness:\s*([\d.]+)')
    nodes_reduction_pattern = re.compile(r'Nodes:\s*(\d+)\s*->\s*(\d+)\s*\(reduced\s*(\d+)\)')
    
    current_level_id = None
    for i, line in enumerate(lines):
        level_match = level_pattern.search(line)
        if level_match:
            current_level_id = int(level_match.group(1))
            
            # Tìm fitness của level này
            for j in range(i+1, min(i+50, len(lines))):
                fitness_match = fitness_pattern.search(lines[j])
                if fitness_match and 'Fitness:' in lines[j] and 'Route details:' not in lines[j-1]:
                    level_fitness = float(fitness_match.group(1))
                    
                    # Tìm node reduction
                    nodes_before = 0
                    nodes_after = 0
                    for k in range(j, min(j+10, len(lines))):
                        reduction_match = nodes_reduction_pattern.search(lines[k])
                        if reduction_match:
                            nodes_before = int(reduction_match.group(1))
                            nodes_after = int(reduction_match.group(2))
                            break
                    
                    level_info.append({
                        'level': current_level_id,
                        'fitness': level_fitness,
                        'nodes_before': nodes_before,
                        'nodes_after': nodes_after
                    })
                    break

    # Parse refinement phase
    refinement_info = []
    refinement_pattern = re.compile(r'=== REFINING FROM LEVEL (\d+) TO LEVEL (\d+) ===')
    unmerged_fitness_pattern = re.compile(r'Unmerged fitness:\s*([\d.]+)')
    after_tabu_pattern = re.compile(r'After tabu:\s*([\d.]+)')
    
    for i, line in enumerate(lines):
        refinement_match = refinement_pattern.search(line)
        if refinement_match:
            from_level = int(refinement_match.group(1))
            to_level = int(refinement_match.group(2))
            
            unmerged_fit = 0.0
            after_tabu_fit = 0.0
            
            for j in range(i+1, min(i+50, len(lines))):
                if 'Unmerged fitness:' in lines[j]:
                    um = unmerged_fitness_pattern.search(lines[j])
                    if um:
                        unmerged_fit = float(um.group(1))
                elif 'After tabu:' in lines[j]:
                    at = after_tabu_pattern.search(lines[j])
                    if at:
                        after_tabu_fit = float(at.group(1))
                        break
            
            refinement_info.append({
                'from_level': from_level,
                'to_level': to_level,
                'unmerged_fitness': unmerged_fit,
                'after_tabu_fitness': after_tabu_fit
            })

    # Tìm section "Route details:" cuối cùng
    vehicle_routes = {}
    last_route_section = -1
    
    for i in range(len(lines) - 1, -1, -1):
        if "Route details:" in lines[i]:
            last_route_section = i
            break
    
    if last_route_section != -1:
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
    
    # Tìm fitness values cuối cùng
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
    
    num_vehicles = len(vehicle_routes)
    num_levels = len(level_info)
    
    result = {
        'Instance': instance_name,
        'Makespan': makespan,
        'DroneViolation': drone_violation,
        'WaitingViolation': waiting_violation,
        'FinalFitness': fitness,
        'IsFeasible': is_feasible,
        'Runtime(s)': round(runtime, 2),
        'NumNodes': num_nodes,
        'SegmentLength': segment_length,
        'NumVehicles': num_vehicles,
        'NumLevels': num_levels,
        'VehicleRoutes': vehicle_routes,
        'LevelInfo': level_info,
        'RefinementInfo': refinement_info
    }
    
    return result

def run_single_instance(cpp_file, instance_file, timeout=600):
    """Chạy 1 instance và return kết quả"""
    
    instance_name = Path(instance_file).stem
    temp_cpp = f"temp_{instance_name}.cpp"
    
    try:
        # Modify C++ file với instance path mới
        with open(cpp_file, 'r', encoding='utf-8') as f:
            cpp_content = f.read()
        
        old_path_pattern = r'read_dataset\(".*?"\);'
        new_path = f'read_dataset("{instance_file}");'
        cpp_content = re.sub(old_path_pattern, new_path, cpp_content)
        
        with open(temp_cpp, 'w', encoding='utf-8') as f:
            f.write(cpp_content)
        
        executable = f"temp_{instance_name}"
        if os.name == 'nt':
            executable += ".exe"
        
        # Compile (không tính thời gian)
        compile_cmd = ["g++", "-O2", "-std=c++17", "-o", executable, temp_cpp]
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            print(f"    ✗ Compilation failed")
            return None
        
        # ✅ BẮT ĐẦU ĐO THỜI GIAN TẠI ĐÂY
        start_time = time.time()
        
        # Run
        if os.name == 'nt':
            run_result = subprocess.run([executable], capture_output=True, text=True, timeout=timeout)
        else:
            run_result = subprocess.run([f"./{executable}"], capture_output=True, text=True, timeout=timeout)
        
        # ✅ KẾT THÚC ĐO THỜI GIAN
        end_time = time.time()
        runtime = end_time - start_time
        
        if run_result.returncode != 0:
            print(f"    ✗ Execution failed")
            return None
        
        result = parse_output(run_result.stdout, instance_name, runtime)
        return result
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timed out")
        return None
    except Exception as e:
        print(f"    ✗ Failed: {e}")
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
    cpp_file = "src/Multilevel_Tabu.cpp"
    instances_dir = "instances"
    results_file = "results/multilevel_results.csv"
    level_details_file = "results/level_details.csv"
    timeout = 600
    
    print("=" * 80)
    print("🚀 MULTILEVEL TABU SEARCH EXPERIMENT")
    print("=" * 80)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 C++ source: {cpp_file}")
    print(f"📁 Instances: {instances_dir}")
    print(f"⏱️  Timeout: {timeout/60:.0f} min/instance")
    print(f"💾 Results: {results_file}")
    print("=" * 80)
    
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists(cpp_file):
        print(f"❌ Error: {cpp_file} not found!")
        return
    
    if not os.path.exists(instances_dir):
        print(f"❌ Error: {instances_dir} not found!")
        return
    
    # Get all instance files
    instance_files = sorted([os.path.join(instances_dir, f) 
                            for f in os.listdir(instances_dir) 
                            if f.endswith('.txt')])
    
    if not instance_files:
        print(f"❌ No .txt files in {instances_dir}")
        return
    
    print(f"\n📋 Found {len(instance_files)} instances\n")
    
    all_results = []
    successful = 0
    failed = 0
    
    exp_start = time.time()
    
    # Run all instances
    for i, instance_file in enumerate(instance_files, 1):
        print(f"[{i}/{len(instance_files)}] 🔄 {Path(instance_file).name}")
        
        result = run_single_instance(cpp_file, instance_file, timeout)
        
        if result:
            all_results.append(result)
            successful += 1
            print(f"   ✅ Runtime: {result['Runtime(s)']:.2f}s, Fitness: {result['FinalFitness']:.4f}")
        else:
            failed += 1
    
    exp_end = time.time()
    total_time = exp_end - exp_start
    
    # Save results
    if all_results:
        max_vehicles = max(r['NumVehicles'] for r in all_results)
        max_levels = max(r['NumLevels'] for r in all_results)
        
        columns = ['Instance', 'FinalFitness', 'Makespan', 'DroneViolation', 
                  'WaitingViolation', 'IsFeasible', 'Runtime(s)', 'NumNodes', 
                  'SegmentLength', 'NumVehicles', 'NumLevels']
        
        for l in range(max_levels):
            columns.append(f'Level{l}_Fitness')
            columns.append(f'Level{l}_NodesReduction')
        
        for v in range(max_vehicles):
            columns.append(f'Vehicle{v}_Route')
        
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in all_results:
                row = {col: result.get(col, "") for col in columns[:11]}
                
                for l in range(max_levels):
                    row[f'Level{l}_Fitness'] = ""
                    row[f'Level{l}_NodesReduction'] = ""
                
                for l, level in enumerate(result.get('LevelInfo', [])):
                    row[f'Level{l}_Fitness'] = level['fitness']
                    row[f'Level{l}_NodesReduction'] = f"{level['nodes_before']}→{level['nodes_after']}"
                
                for v in range(max_vehicles):
                    row[f'Vehicle{v}_Route'] = ""
                
                vehicle_routes = result.get('VehicleRoutes', {})
                for v_id, route_str in vehicle_routes.items():
                    if v_id < max_vehicles:
                        row[f'Vehicle{v_id}_Route'] = f'"{route_str}"'
                
                writer.writerow(row)
        
        print(f"\n✅ Results saved: {results_file}")
        
        # Save level details
        with open(level_details_file, 'w', newline='', encoding='utf-8') as f:
            detail_cols = ['Instance', 'Phase', 'FromLevel', 'ToLevel', 
                          'Fitness', 'NodesReduction', 'Notes']
            writer = csv.DictWriter(f, fieldnames=detail_cols)
            writer.writeheader()
            
            for result in all_results:
                instance = result['Instance']
                
                for level in result.get('LevelInfo', []):
                    writer.writerow({
                        'Instance': instance,
                        'Phase': 'Coarsening',
                        'FromLevel': level['level'],
                        'ToLevel': level['level'] + 1,
                        'Fitness': level['fitness'],
                        'NodesReduction': f"{level['nodes_before']}→{level['nodes_after']}" if level['nodes_before'] > 0 else "",
                        'Notes': 'After tabu'
                    })
                
                for ref in result.get('RefinementInfo', []):
                    writer.writerow({
                        'Instance': instance,
                        'Phase': 'Refinement',
                        'FromLevel': ref['from_level'],
                        'ToLevel': ref['to_level'],
                        'Fitness': ref['unmerged_fitness'],
                        'NodesReduction': '',
                        'Notes': 'Unmerged'
                    })
                    writer.writerow({
                        'Instance': instance,
                        'Phase': 'Refinement',
                        'FromLevel': ref['from_level'],
                        'ToLevel': ref['to_level'],
                        'Fitness': ref['after_tabu_fitness'],
                        'NodesReduction': '',
                        'Notes': 'After tabu'
                    })
        
        print(f"✅ Details saved: {level_details_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"🎯 SUMMARY")
    print(f"{'='*80}")
    print(f"📁 Total: {len(instance_files)}")
    print(f"✅ Success: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time/60:.1f} min")
    print(f"⏱️  Avg per instance: {total_time/len(instance_files):.1f}s")
    
    if all_results:
        fitnesses = [r['FinalFitness'] for r in all_results]
        runtimes = [r['Runtime(s)'] for r in all_results]
        feasible = sum(1 for r in all_results if r['IsFeasible'])
        
        print(f"\n📈 STATISTICS:")
        print(f"✔️  Feasible: {feasible}/{len(all_results)} ({feasible/len(all_results)*100:.1f}%)")
        print(f"🏆 Best fitness: {min(fitnesses):.4f}")
        print(f"📉 Worst fitness: {max(fitnesses):.4f}")
        print(f"📊 Avg fitness: {sum(fitnesses)/len(fitnesses):.4f}")
        print(f"⏱️  Avg runtime: {sum(runtimes)/len(runtimes):.2f}s")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()