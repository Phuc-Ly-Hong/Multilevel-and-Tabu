#include<bits/stdc++.h>
using namespace std;

struct Node {
    int id;
    double x,y;
    double c1_or_c2;
    double limit_wait = 60.0; // (phút)
};

struct VehicleFamily {
    int id;
    double speed;
    bool is_drone;
    double limit_drone; // (m/phút)
};

struct Solution {
    vector<vector<int>> route; // danh sách các khách hàng trong route
    double makespan; // thời gian hoàn thành
    double drone_violation; // tổng số thời gian vi phạm thời gian bay của drone
    double waiting_violation; // tổng số thời gian vi phạm chờ tối đa
    double fitness; // giá trị hàm mục tiêu
    bool is_feasible; // lời giải có hợp lệ không

    Solution(): makespan(0), drone_violation(0), waiting_violation(0), fitness(DBL_MAX), is_feasible(true) {}
};

struct TabuMove {
    string type; // 1-0, 1-1, 2-0, 2-1, 2-2, 2-opt
    int customer_id1; // khách hàng thứ nhất được di chuyển của xe 1
    int customer_id2; // khách hàng thứ hai được di chuyển của xe 1
    int customer_id3; // khách hàng thứ nhất được di chuyển của xe 2
    int customer_id4; // khách hàng thứ hai được di chuyển của xe 2
    int vehicle1; // từ xe nào
    int vehicle2; // đến xe nào
    int pos1; // vị trí trong route của xe từ
    int pos2; // vị trí trong route của xe từ (thứ 2)
    int pos3; // vị trí trong route của xe đến
    int pos4; // vị trí trong route của xe đến (thứ 2)
    int tenure; // số vòng lặp còn lại move này bị tabu
};

struct RouteAnalysis {
    vector<double> cumulative_flight_time;  // thời gian bay tích lũy từ depot gần nhất
    vector<double> arrival_time;            // thời gian đến từng điểm
    vector<double> waiting_times;           // thời gian chờ tại mỗi khách hàng
    double total_flight_time;               // tổng thời gian bay từ depot cuối
    double total_waiting;                   // tổng vi phạm waiting
    double max_waiting;                     // vi phạm waiting lớn nhất
    
    RouteAnalysis() : total_flight_time(0), total_waiting(0), max_waiting(0) {}
};

struct LevelInfo {
    vector<Node> nodes;
    vector<Node> C1_level, C2_level; // customers ở level này
    map<int, vector<int>> node_mapping; // ánh xạ từ node level này về node gốc
    int level_id;
    int num_customers;

    LevelInfo() : level_id(0), num_customers(0) {}
};

struct MergedNodeOrientation {
    int merged_node_id;      // ID của merged node
    bool is_reversed;        // Có đảo chiều không
    int level_id;            // Level nào
    
    MergedNodeOrientation(int id = -1, bool rev = false, int lvl = -1) 
        : merged_node_id(id), is_reversed(rev), level_id(lvl) {}
};

vector<vector<double>> distances;
vector<vector<double>> original_distances; // dùng khi merge các khách hàng cho level
vector<Node> C1; // customers served only by technicians
vector<Node> C2; // customers served by drones or technicians
vector<VehicleFamily> vehicles;
map<int, MergedNodeOrientation> merged_node_orientations;

int depot_id = 0;
int num_nodes = 0;
double alpha1 = 1.0; // tham số hàm phạt thứ nhất
double alpha2 = 1.0; // tham số hàm phạt thứ hai
double Beta = 0.5; // tham số điều chỉnh hệ số hàm phạt

int MAX_ITER;
int TABU_TENURE;
int MAX_NO_IMPROVE = 4000;
double EPSILON = 1e-6;

// Adaptive parameters
int SEGMENT_LENGTH;
vector<string> MOVE_SET = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
vector<double> scorePi = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
vector<double> used_count = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

const double delta1 = 0.3;
const double delta2 = 0.2;
const double delta3 = 0.1;
const double delta4 = 0.3;

int select_move_type(){
    double total_weight = accumulate(weights.begin(), weights.end(), 0.0);
    double r = ((double)rand() / RAND_MAX) * total_weight;
    double cumulate = 0.0;

    for (size_t i = 0; i < MOVE_SET.size(); i++){
        cumulate += weights[i];
        if (r <= cumulate) return i;
    }
    return MOVE_SET.size() - 1; 
}

void update_weights(){
    for (int i = 0; i < MOVE_SET.size(); i++){
        if (used_count[i] > 0){
            double avg_score = scorePi[i] / used_count[i];
            weights[i] = (1.0 - delta4) * weights[i] + delta4 * avg_score;
        }
        scorePi[i] = 0.0;
        used_count[i] = 0.0;
    }
}

void read_dataset(const string &filename){
    vector<Node> nodes;
    ifstream file(filename);
    if (!file.is_open()){
        cerr << "Error opening file: " << filename <<endl;
        exit(1);
    }
    nodes.push_back({depot_id,0.0,0.0,-1.0,DBL_MAX}); // depot
    string line;
    while (getline(file,line)){
        if (line.empty() || line[0] == '#'|| isalpha(line[0])) continue;
        istringstream ss(line);
        double demand;
        double x,y;
        static int id = 1;
        ss >> x >> y >> demand;
        nodes.push_back({id++,x,y,demand});
    }
    file.close();

    cout << "Read " << nodes.size() << " nodes (including depot)." << endl;
    if (nodes.size() >= 100) {
        MAX_ITER = 80 * nodes.size() / 2;
        SEGMENT_LENGTH = 75;
    } else if (nodes.size() >= 50){
        MAX_ITER = 3000;
        SEGMENT_LENGTH = 60;
    } else {
        MAX_ITER = 2000;
        SEGMENT_LENGTH = 50;
    }
    for (const auto& node : nodes) {
        if (node.id == depot_id) {
            cout << "Node id: " << node.id << " (depot), x: " << node.x << ", y: " << node.y << endl;
            continue;
        } else {
            cout << "Node id: " << node.id << ", x: " << node.x << ", y: " << node.y
                 << ", type: " << (node.c1_or_c2 > 0 ? "C2" : "C1") << ", limit_wait: " << node.limit_wait << endl;
        }
    }

    // Tính toán khoảng cách giữa các nút
    distances.resize(nodes.size(), vector<double>(nodes.size(), 0));
    for (size_t i = 0; i < nodes.size(); ++i){
        for (size_t j = 0; j < nodes.size(); ++j){
            if (i != j){
                distances[i][j] = sqrt(pow(nodes[i].x - nodes[j].x, 2) + pow(nodes[i].y - nodes[j].y, 2));
            }
        }
    }

    // Phân loại khách hàng
    for (const auto& node : nodes){
        if (node.id == depot_id) continue;
        if (node.c1_or_c2 > 0){
            C2.push_back(node);
        } else if (node.c1_or_c2 == 0) {
            C1.push_back(node);
        }
    }
    cout << "C1 size: " << C1.size() << ", C2 size: " << C2.size() << endl;
    num_nodes = nodes.size();
    TABU_TENURE = min((int)ceil(num_nodes/4.0), 10);
}

void print_solution(const Solution &sol){
    cout << "Route details:" << endl;
    for (size_t v = 0; v < sol.route.size(); v++) {
        cout << "Vehicle " << v << ": ";
        for (int cid : sol.route[v]) cout << cid << " ";
        cout << endl;
    }
    cout << "Makespan: " << sol.makespan << endl;
    cout << "Drone violation: " << sol.drone_violation << endl;
    cout << "Waiting violation: " << sol.waiting_violation << endl;
    cout << "Fitness: " << sol.fitness << endl;
}

map <int, int> node_id_to_index_cache;

void update_node_index_cache(const LevelInfo& level) {
    node_id_to_index_cache.clear();
    for (size_t i = 0; i < level.nodes.size(); i++) {
        node_id_to_index_cache[level.nodes[i].id] = i; 
    }
}

int find_node_index_fast(int node_id) {
    auto it = node_id_to_index_cache.find(node_id);
    if (it != node_id_to_index_cache.end()) {
        return it->second;  
    }
    return -1;
}

void evaluate_solution(Solution &sol, const LevelInfo *current_level = nullptr) {
    sol.makespan = 0;
    sol.drone_violation = 0;
    sol.waiting_violation = 0;
    sol.fitness = 0;
    sol.is_feasible = true;

    for (size_t i = 0; i < sol.route.size(); i++){
        int prev = depot_id;
        double current_time = 0;
        double depart_time = 0;
        vector<pair<int, double>> served_in_trip;

        for (int j = 0; j < sol.route[i].size(); j++) {
            int cid = sol.route[i][j];
            
            if (cid == depot_id){
                if (prev != depot_id){
                    double travel_distance = 0.0;
                    
                    if (current_level != nullptr) {
                        int prev_idx = find_node_index_fast(prev);
                        if (prev_idx == -1) {
                            cerr << "ERROR: Cannot find node " << prev << " in level " 
                                << current_level->level_id << endl;
                            sol.fitness = DBL_MAX;
                            return;
                        }
                        int depot_idx = 0; 
                        
                        if (prev_idx >= 0 && prev_idx < current_level->nodes.size()) {
                            auto it = current_level->node_mapping.find(prev);
                            if (it != current_level->node_mapping.end() && it->second.size() > 1) {
                                if (prev_idx < original_distances.size() && 
                                    depot_idx < original_distances[0].size()) {
                                    travel_distance = original_distances[prev_idx][depot_idx];
                                } else {
                                    cerr << "ERROR: original_distances bounds - prev_idx=" 
                                         << prev_idx << ", depot_idx=" << depot_idx 
                                         << ", size=" << original_distances.size() << "x" 
                                         << (original_distances.empty() ? 0 : original_distances[0].size()) << endl;
                                }
                            } else {
                                if (prev_idx < distances.size() && 
                                    depot_idx < distances[0].size()) {
                                    travel_distance = distances[prev_idx][depot_idx];
                                } else {
                                    cerr << "ERROR: distances bounds - prev_idx=" 
                                         << prev_idx << ", depot_idx=" << depot_idx 
                                         << ", size=" << distances.size() << "x" 
                                         << (distances.empty() ? 0 : distances[0].size()) << endl;
                                }
                            }
                        } else {
                            cerr << "ERROR: prev_idx=" << prev_idx 
                                 << " out of bounds for node " << prev 
                                 << " (level size=" << current_level->nodes.size() << ")" << endl;
                        }
                    } else {
                        if (prev < distances.size() && depot_id < distances[0].size()) {
                            travel_distance = distances[prev][depot_id];
                        }
                    }
                    
                    current_time += travel_distance / vehicles[i].speed;
                }
                
                double arrival_depot = current_time;
                double flight_time = arrival_depot - depart_time;
                
                if (vehicles[i].is_drone && flight_time > vehicles[i].limit_drone){
                    sol.drone_violation += (flight_time - vehicles[i].limit_drone);
                }
                
                for (auto &p : served_in_trip){
                    double time_served = p.second;
                    double wait_time = arrival_depot - time_served;
                    if (!C2.empty() && wait_time > C2[0].limit_wait) {
                        sol.waiting_violation += (wait_time - C2[0].limit_wait);
                    }
                }
                
                if (sol.drone_violation > 0 || sol.waiting_violation > 0) {
                    sol.is_feasible = false;
                }
                
                depart_time = current_time;
                served_in_trip.clear();
                prev = depot_id;
            }
            else {
                double travel_distance = 0.0;
                
                if (current_level != nullptr) {
                    int prev_idx = find_node_index_fast(prev);
                    int cid_idx = find_node_index_fast(cid);

                    if (prev_idx >= 0 && cid_idx >= 0 && prev_idx < current_level->nodes.size() && cid_idx < current_level->nodes.size()) {
                        
                        auto it_prev = current_level->node_mapping.find(prev);
                        auto it_cid = current_level->node_mapping.find(cid);
                        
                        bool prev_is_merged = (it_prev != current_level->node_mapping.end() && it_prev->second.size() > 1);
                        bool cid_is_merged = (it_cid != current_level->node_mapping.end() && it_cid->second.size() > 1);

                        if (prev_is_merged || cid_is_merged) {
                            if (prev_idx < original_distances.size() && cid_idx < original_distances[0].size()) {
                                travel_distance = original_distances[prev_idx][cid_idx];
                            } else {
                                cerr << "ERROR: original_distances bounds - prev=" << prev 
                                     << " (idx=" << prev_idx << "), cid=" << cid 
                                     << " (idx=" << cid_idx << "), size=" 
                                     << original_distances.size() << "x" 
                                     << (original_distances.empty() ? 0 : original_distances[0].size()) << endl;
                            }
                        } else {
                            if (prev_idx < distances.size() && cid_idx < distances[0].size()) {
                                travel_distance = distances[prev_idx][cid_idx];
                            } else {
                                cerr << "ERROR: distances bounds - prev=" << prev 
                                     << " (idx=" << prev_idx << "), cid=" << cid 
                                     << " (idx=" << cid_idx << "), size=" 
                                     << distances.size() << "x" 
                                     << (distances.empty() ? 0 : distances[0].size()) << endl;
                            }
                        }
                    } else {
                        cerr << "ERROR: Invalid indices - prev=" << prev 
                             << " (idx=" << prev_idx << "), cid=" << cid 
                             << " (idx=" << cid_idx << "), level size=" 
                             << current_level->nodes.size() << endl;
                    }
                } else {
                    if (prev < distances.size() && cid < distances[0].size()) {
                        travel_distance = distances[prev][cid];
                    }
                }
                
                current_time += travel_distance / vehicles[i].speed;
                served_in_trip.push_back({cid, current_time});
                prev = cid;
            }
        }
        sol.makespan = max(sol.makespan, current_time);
    }

    sol.fitness = sol.makespan + alpha1*sol.drone_violation + alpha2*sol.waiting_violation;
}

Solution init_greedy_solution() {
    Solution sol;
    sol.route.resize(vehicles.size());

    for (size_t v = 0; v < vehicles.size(); ++v)
        sol.route[v].push_back(depot_id);

    /// ✅ 1. GÁN C1 CHO TECHNICIAN (GIỮ NGUYÊN)
    vector<int> unserved_C1;
    for (const auto& n : C1) unserved_C1.push_back(n.id);

    vector<int> tech_pos;
    vector<int> tech_indices;
    for (size_t v = 0; v < vehicles.size(); v++) {
        if (!vehicles[v].is_drone) {
            tech_pos.push_back(depot_id);
            tech_indices.push_back(v);
        }
    }

    while (!unserved_C1.empty()) {
        double best_dist = DBL_MAX;
        int best_tech = -1, best_cid = -1, best_idx = -1;
        for (size_t t = 0; t < tech_indices.size(); t++) {
            for (size_t i = 0; i < unserved_C1.size(); i++) {
                int cid = unserved_C1[i];
                double d = distances[tech_pos[t]][cid];
                if (d < best_dist) {
                    best_dist = d;
                    best_tech = t;
                    best_cid = cid;
                    best_idx = i;
                }
            }
        }
        int vehicle_id = tech_indices[best_tech];
        sol.route[vehicle_id].push_back(best_cid);
        tech_pos[best_tech] = best_cid;
        unserved_C1.erase(unserved_C1.begin() + best_idx);
    } 

    /// ✅ 2. GÁN C2 - BALANCED ALLOCATION
    vector<int> unserved_C2;
    for (const auto& n : C2) unserved_C2.push_back(n.id);

    vector<int> current_pos(vehicles.size(), depot_id);
    for (size_t v = 0; v < vehicles.size(); v++) {
        if (sol.route[v].size() > 1) {
            current_pos[v] = sol.route[v].back();
        }
    }

    // ✅ TÍNH CUSTOMERS PER VEHICLE (ĐỀU)
    int total_vehicles = vehicles.size();
    int target_per_vehicle = (unserved_C2.size() + total_vehicles - 1) / total_vehicles;
    
    cout << "\n📊 TARGET ALLOCATION:" << endl;
    cout << "  Total C2 customers: " << unserved_C2.size() << endl;
    cout << "  Total vehicles: " << total_vehicles << endl;
    cout << "  Target per vehicle: " << target_per_vehicle << endl;

    // ✅ GÁN ĐỀU CHO MỖI XE
    while (!unserved_C2.empty()) {
        for (size_t v = 0; v < vehicles.size() && !unserved_C2.empty(); v++) {
            // Đếm customers hiện tại
            int current_count = 0;
            for (int node : sol.route[v]) {
                if (node != depot_id) current_count++;
            }
            
            // ✅ CHỈ GÁN NẾU XE CHƯA ĐỦ TARGET
            if (current_count >= target_per_vehicle) {
                continue;
            }
            
            // Tìm nearest customer
            double best_dist = DBL_MAX;
            int best_idx = -1;
            for (size_t i = 0; i < unserved_C2.size(); i++) {
                int cid = unserved_C2[i];
                double d = distances[current_pos[v]][cid];
                if (d < best_dist) {
                    best_dist = d;
                    best_idx = i;
                }
            }
            
            if (best_idx != -1) {
                int best_cid = unserved_C2[best_idx];
                sol.route[v].push_back(best_cid);
                current_pos[v] = best_cid;
                unserved_C2.erase(unserved_C2.begin() + best_idx);
                
                cout << "  Vehicle " << v << " ← customer " << best_cid 
                     << " (now has " << (current_count + 1) << " customers)" << endl;
            }
        }
    }

    // ✅ GÁN CUSTOMERS THỪA (NẾU CÓ)
    while (!unserved_C2.empty()) {
        double best_dist = DBL_MAX;
        int best_v = -1, best_idx = -1;
        
        for (size_t v = 0; v < vehicles.size(); v++) {
            for (size_t i = 0; i < unserved_C2.size(); i++) {
                int cid = unserved_C2[i];
                double d = distances[current_pos[v]][cid];
                if (d < best_dist) {
                    best_dist = d;
                    best_v = v;
                    best_idx = i;
                }
            }
        }
        
        if (best_idx != -1) {
            int best_cid = unserved_C2[best_idx];
            sol.route[best_v].push_back(best_cid);
            current_pos[best_v] = best_cid;
            unserved_C2.erase(unserved_C2.begin() + best_idx);
        }
    }

    // Kết thúc routes
    for (size_t v = 0; v < vehicles.size(); v++) {
        if (sol.route[v].empty() || sol.route[v].back() != depot_id) {
            sol.route[v].push_back(depot_id);
        }
    }

    evaluate_solution(sol);
    
    // ✅ HIỂN THỊ WORKLOAD
    cout << "\n📊 INITIAL WORKLOAD:" << endl;
    for (size_t v = 0; v < vehicles.size(); v++) {
        int count = 0;
        for (int node : sol.route[v]) {
            if (node != depot_id) count++;
        }
        cout << "  Vehicle " << v << ": " << count << " customers";
        if (abs(count - target_per_vehicle) > 2) {
            cout << " ⚠️  IMBALANCED";
        } else {
            cout << " ✅";
        }
        cout << endl;
    }
    
    return sol;
}

int get_type(int nid, const LevelInfo *current_level = nullptr) {
    if (current_level != nullptr) {
        // Dùng level hiện tại
        for (const auto& n : current_level->C2_level) {
            if (n.id == nid) return 2;
        }
        for (const auto& n : current_level->C1_level) {
            if (n.id == nid) return 1;
        }
    } else {
        // Dùng global C1, C2
        for (const auto& n : C2) if (n.id == nid) return 2;
        for (const auto& n : C1) if (n.id == nid) return 1;
    }
    return -1;
}

RouteAnalysis analyze_drone_route(const vector<int> &route, int vehicle_idx, const LevelInfo *current_level = nullptr) {
    RouteAnalysis analysis;
    
    if (route.size() <= 2) {
        return analysis;
    }
    
    analysis.cumulative_flight_time.resize(route.size(), 0.0);
    analysis.arrival_time.resize(route.size(), 0.0);
    analysis.waiting_times.resize(route.size(), 0.0);
    
    double current_time = 0.0;
    double flight_time_since_last_depot = 0.0;
    double max_flight_segment = 0.0;
    int last_node = depot_id;
    
    vector<pair<int, double>> current_trip_customers;
    double trip_start_time = 0.0;
    
    for (size_t i = 0; i < route.size(); i++) {
        int current_node = route[i];
        
        if (i > 0) {
            double travel_distance = 0.0;
            
            if (current_level != nullptr) {
                int last_idx = find_node_index_fast(last_node);
                int curr_idx = find_node_index_fast(current_node);
                
                if (last_idx == -1 || curr_idx == -1) {
                    cerr << "ERROR in analyze_drone_route: Cannot find nodes " 
                         << last_node << " or " << current_node << endl;
                    return analysis; 
                }
                
                if (last_idx >= distances.size() || curr_idx >= distances[0].size()) {
                    cerr << "ERROR: Matrix bounds in analyze_drone_route" << endl;
                    return analysis;
                }
                
                // Kiểm tra merged nodes
                auto it_last = current_level->node_mapping.find(last_node);
                auto it_curr = current_level->node_mapping.find(current_node);
                
                bool last_is_merged = (it_last != current_level->node_mapping.end() && 
                                      it_last->second.size() > 1);
                bool curr_is_merged = (it_curr != current_level->node_mapping.end() && 
                                      it_curr->second.size() > 1);
                
                if (last_is_merged || curr_is_merged) {
                    travel_distance = original_distances[last_idx][curr_idx];
                } else {
                    travel_distance = distances[last_idx][curr_idx];
                }
            } else {
                // Level 0 - dùng node ID trực tiếp
                if (last_node < distances.size() && current_node < distances[0].size()) {
                    travel_distance = distances[last_node][current_node];
                }
            }
            
            double travel_time = travel_distance / vehicles[vehicle_idx].speed;
            current_time += travel_time;
            
            if (current_node == depot_id) {
                double trip_end_time = current_time;
                
                for (auto &p : current_trip_customers) {
                    int idx = p.first;
                    double service_time = p.second;
                    double wait_time = trip_end_time - service_time;
                    
                    analysis.waiting_times[idx] = wait_time;
                    
                    if (wait_time > 60.0) {
                        double violation = wait_time - 60.0;
                        analysis.total_waiting += violation;
                        analysis.max_waiting = max(analysis.max_waiting, wait_time);
                    }
                }
                
                max_flight_segment = max(max_flight_segment, flight_time_since_last_depot);
                flight_time_since_last_depot = 0.0;
                current_trip_customers.clear();
                trip_start_time = current_time;
            } else {
                flight_time_since_last_depot += travel_time;
                current_trip_customers.push_back({i, current_time});
            }
        }
        
        analysis.cumulative_flight_time[i] = flight_time_since_last_depot;
        analysis.arrival_time[i] = current_time;
        last_node = current_node;
    }
    
    analysis.total_flight_time = max_flight_segment;
    
    return analysis;
}

// Tìm vị trí tốt nhất để chèn depot vào route
int find_best_depot_insertion(const vector<int> &route, int vehicle_idx, const LevelInfo *current_level = nullptr) {
    if (!vehicles[vehicle_idx].is_drone || route.size() <= 3) {
        return -1;
    }

    RouteAnalysis original = analyze_drone_route(route, vehicle_idx, current_level);
    
    double original_violation = 0.0;
    if (original.total_flight_time > vehicles[vehicle_idx].limit_drone) {
        original_violation += alpha1 * (original.total_flight_time - vehicles[vehicle_idx].limit_drone);
    }
    original_violation += alpha2 * original.total_waiting;

    double best_improvement = -DBL_MAX;  // ✅ CHO PHÉP ÂM
    int best_pos = -1;
    
    for (size_t pos = 2; pos < route.size() - 1; pos++) {
        if (route[pos - 1] == depot_id || route[pos] == depot_id) {
            continue;
        }
        
        vector<int> test_route = route;
        test_route.insert(test_route.begin() + pos, depot_id);
        
        RouteAnalysis test_analysis = analyze_drone_route(test_route, vehicle_idx, current_level);
        
        double test_violation = 0.0;
        if (test_analysis.total_flight_time > vehicles[vehicle_idx].limit_drone) {
            test_violation += alpha1 * (test_analysis.total_flight_time - vehicles[vehicle_idx].limit_drone);
        }
        test_violation += alpha2 * test_analysis.total_waiting;
        
        double improvement = original_violation - test_violation;
        
        // Tính detour penalty NHỎ HƠN
        double detour_distance = 0.0;
        if (current_level != nullptr) {
            int idx_prev = find_node_index_fast(route[pos - 1]);
            int idx_depot = 0;
            int idx_curr = find_node_index_fast(route[pos]);

            if (idx_prev != -1 && idx_curr != -1) {
                detour_distance = distances[idx_prev][idx_depot] + distances[idx_depot][idx_curr] - distances[idx_prev][idx_curr];
            }
        } else {
            detour_distance = distances[route[pos - 1]][depot_id] + distances[depot_id][route[pos]] - distances[route[pos - 1]][route[pos]];
        }
        
        double detour_penalty = 0.01 * detour_distance;  // ✅ GIẢM TỪ 0.05 → 0.01
        improvement -= detour_penalty;
        
        if (improvement > best_improvement) {
            best_improvement = improvement;
            best_pos = pos;
        }
    }
    
    // ✅ CHẤP NHẬN NẾU CẢI THIỆN > 0.01 (thay vì 0.5)
    return (best_improvement > 0.01) ? best_pos : -1;
}

void optimize_all_drone_routes(Solution &sol, const LevelInfo *current_level = nullptr) {
    bool changed = true;
    int max_rounds = 3;
    int round = 0;
    
    while (changed && round < max_rounds) {
        changed = false;
        round++;
        
        for (size_t v = 0; v < vehicles.size(); v++) {
            if (!vehicles[v].is_drone) continue;
            
            int insert_pos = find_best_depot_insertion(sol.route[v], v, current_level);
            
            if (insert_pos != -1) {
                bool has_adjacent_depot = false;
                if (insert_pos > 0 && sol.route[v][insert_pos - 1]  == depot_id) {
                    has_adjacent_depot = true;
                }
                if (insert_pos < sol.route[v].size() && sol.route[v][insert_pos] == depot_id) {
                    has_adjacent_depot = true;
                }
                if (!has_adjacent_depot) {
                    sol.route[v].insert(sol.route[v].begin() + insert_pos, depot_id);
                    evaluate_solution(sol, current_level);
                    changed = true;
                }
            }
        }
    }
}

void remove_redundant_depots(Solution &sol, const LevelInfo *current_level = nullptr) {
    bool changed = true;
    int round = 0;
    const int MAX_ROUNDS = 5;
    
    while (changed && round < MAX_ROUNDS) {
        changed = false;
        round++;
        
        for (size_t v = 0; v < vehicles.size(); v++) {
            if (!vehicles[v].is_drone) continue;
            
            vector<int> &route = sol.route[v];
            bool found_duplicate = false;
            
            for (size_t i = 1; i < route.size() - 1 && !found_duplicate; ) {
                if (i + 1 < route.size() && route[i] == depot_id && route[i + 1] == depot_id) {
                    route.erase(route.begin() + i);
                    found_duplicate = true;
                    changed = true;
                } else {
                    i++;
                }
            }
            
            if (found_duplicate) {
                continue;
            }
            
            bool removed_redundant = false;
            for (size_t i = 1; i < route.size() - 1 && !removed_redundant; ) {
                if (route[i] == depot_id) {
                    vector<int> test_route = route;
                    test_route.erase(test_route.begin() + i);
                    
                    Solution test_sol = sol;
                    test_sol.route[v] = test_route;
                    evaluate_solution(test_sol, current_level);
                    
                    if (test_sol.fitness <= sol.fitness + EPSILON) {
                        route = test_route;
                        sol = test_sol;
                        changed = true;
                        removed_redundant = true;
                    } else {
                        i++;
                    }
                } else {
                    i++;
                }
            }
        }
    }
    
    if (round >= MAX_ROUNDS) {
        cout << "⚠️  remove_redundant_depots reached max rounds" << endl;
    }
}

map<pair<int,int>, int> edge_frequency;

void update_edge_frequency(const Solution& best_solution) {
    for (size_t v = 0; v < best_solution.route.size(); v++) {
        const vector<int>& route = best_solution.route[v];
        for (size_t i = 0; i < route.size() - 1; i++) {
            int from_node = route[i];
            int to_node = route[i + 1];
            if (from_node != depot_id && to_node != depot_id) {
                pair<int,int> edge = make_pair(from_node, to_node);
                edge_frequency[edge]++;
                //cout << "Edge (" << from_node << ", " << to_node << ") frequency: " << edge_frequency[edge] << endl;
            }
        }
    }
}

bool is_merged_node(int node_id, const LevelInfo& level) {
    if (node_id == depot_id) return false;
    auto it = level.node_mapping.find(node_id);
    return (it != level.node_mapping.end() && it->second.size() > 1);
}

vector<int> get_merged_group(int node_id, const LevelInfo& level) {
    auto it = level.node_mapping.find(node_id);
    if (it != level.node_mapping.end()) {
        return it->second;
    }
    return {node_id};
}

double calculate_orientation_cost(int prev_node, int next_node, const vector<int>& group, bool reverse_group, const LevelInfo* level) {
    if (group.size() <= 1) return 0.0;
    
    // KIỂM TRA LEVEL
    if (level == nullptr) {
        cerr << "ERROR: level is null in calculate_orientation_cost" << endl;
        return DBL_MAX;
    }
    
    // Entry và exit nodes dựa trên orientation
    int entry_node = reverse_group ? group.back() : group.front();
    int exit_node = reverse_group ? group.front() : group.back();
    
    // TÌM INDEX CHO TẤT CẢ NODES - BỎ QUA NẾU KHÔNG TÌM THẤY
    int idx_prev = find_node_index_fast(prev_node);
    int idx_entry = find_node_index_fast(entry_node);
    int idx_exit = find_node_index_fast(exit_node);
    int idx_next = find_node_index_fast(next_node);
    
    // KIỂM TRA CÁC TRƯỜNG HỢP LỖI
    if (idx_prev < 0 || idx_entry < 0 || idx_exit < 0 || idx_next < 0) {
        // Không tìm thấy node trong level hiện tại - SKIP
        return DBL_MAX;
    }
    
    if (idx_prev >= distances.size() || idx_next >= distances.size() ||
        idx_entry >= distances.size() || idx_exit >= distances.size()) {
        cerr << "ERROR: Index out of bounds in calculate_orientation_cost" << endl;
        cerr << "  prev=" << idx_prev << ", entry=" << idx_entry 
             << ", exit=" << idx_exit << ", next=" << idx_next 
             << ", matrix_size=" << distances.size() << endl;
        return DBL_MAX;
    }
    
    if (distances.empty() || distances[0].size() < distances.size()) {
        cerr << "ERROR: Invalid distance matrix dimensions" << endl;
        return DBL_MAX;
    }
    
    double cost = 0.0;
    
    // KHOẢNG CÁCH TỪ PREV ĐẾN ENTRY
    if (idx_prev < distances.size() && idx_entry < distances[0].size()) {
        cost += distances[idx_prev][idx_entry];
    } else {
        return DBL_MAX;
    }
    
    // KHOẢNG CÁCH TRONG GROUP
    if (reverse_group) {
        // Đảo chiều: group.back() -> ... -> group.front()
        for (int i = group.size() - 1; i > 0; i--) {
            int from_idx = find_node_index_fast(group[i]);
            int to_idx = find_node_index_fast(group[i - 1]);
            
            if (from_idx < 0 || to_idx < 0) {
                // Node trong group không tồn tại ở level hiện tại
                // Đây là trường hợp node gốc bị merge
                return DBL_MAX;
            }
            
            if (from_idx >= distances.size() || to_idx >= distances[0].size()) {
                return DBL_MAX;
            }
            
            cost += distances[from_idx][to_idx];
        }
    } else {
        // Thuận: group.front() -> ... -> group.back()
        for (size_t i = 0; i < group.size() - 1; i++) {
            int from_idx = find_node_index_fast(group[i]);
            int to_idx = find_node_index_fast(group[i + 1]);
            
            if (from_idx < 0 || to_idx < 0) {
                // Node trong group không tồn tại ở level hiện tại
                return DBL_MAX;
            }
            
            if (from_idx >= distances.size() || to_idx >= distances[0].size()) {
                return DBL_MAX;
            }
            
            cost += distances[from_idx][to_idx];
        }
    }
    
    // KHOẢNG CÁCH TỪ EXIT ĐẾN NEXT
    if (idx_exit < distances.size() && idx_next < distances[0].size()) {
        cost += distances[idx_exit][idx_next];
    } else {
        return DBL_MAX;
    }
    
    return cost;
}

pair<bool, double> find_best_orientation(const vector<int>& route, int pos, int merged_node_id, const LevelInfo* level) {
    if (level == nullptr) {
        return {false, 0.0};
    }
    
    if (!is_merged_node(merged_node_id, *level)) {
        return {false, 0.0};
    }
    
    vector<int> group = get_merged_group(merged_node_id, *level);
    if (group.size() <= 1) {
        return {false, 0.0};
    }
    
    int prev_node = depot_id;
    int next_node = depot_id;
    
    if (pos > 0 && pos < route.size()) {
        prev_node = route[pos - 1];
    }
    if (pos < route.size() - 1) {
        next_node = route[pos + 1];
    }
    
    double cost_normal = calculate_orientation_cost(prev_node, next_node, group, false, level);
    double cost_reversed = calculate_orientation_cost(prev_node, next_node, group, true, level);
    
    if (cost_normal >= DBL_MAX - 1.0 && cost_reversed >= DBL_MAX - 1.0) {
        return {false, 0.0};
    }
    
    bool should_reverse = false;
    double chosen_cost = 0.0;
    
    if (cost_normal < DBL_MAX - 1.0 || cost_reversed < DBL_MAX - 1.0) {
        if (cost_reversed < cost_normal - EPSILON) {
            should_reverse = true;
            chosen_cost = cost_reversed;
            
            // ✅ LƯU ORIENTATION INFO
            merged_node_orientations[merged_node_id] = 
                MergedNodeOrientation(merged_node_id, true, level->level_id);
            
            cout << "    Orientation: node=" << merged_node_id 
                 << " normal=" << cost_normal 
                 << " reversed=" << cost_reversed << " → REVERSED ✓" << endl;
        } else {
            should_reverse = false;
            chosen_cost = cost_normal;
            
            // ✅ LƯU ORIENTATION INFO (NORMAL)
            merged_node_orientations[merged_node_id] = 
                MergedNodeOrientation(merged_node_id, false, level->level_id);
            
            cout << "    Orientation: node=" << merged_node_id 
                 << " normal=" << cost_normal 
                 << " reversed=" << cost_reversed << " → NORMAL" << endl;
        }
    }
    
    return {should_reverse, chosen_cost};
}

bool contains_depot_in_range(const vector<int>& route, size_t start, size_t end) {
    for (size_t i = start; i <= end && i < route.size(); i++) {
        if (route[i] == depot_id) return true;
    }
    return false;
}

bool is_tabu(const vector<TabuMove> &tabu_list, const TabuMove &move){
    for (const auto &tabu_move : tabu_list){
        if (tabu_move.type == move.type && tabu_move.tenure > 0){
            if (move.type == "1-1"){
                if (((tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id3 == move.customer_id3) ||
                     (tabu_move.customer_id1 == move.customer_id3 && tabu_move.customer_id3 == move.customer_id1)) &&
                    ((tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2) ||
                     (tabu_move.vehicle1 == move.vehicle2 && tabu_move.vehicle2 == move.vehicle1))) {
                    return true;
                }
            }
            else if (move.type == "1-0"){
                if (tabu_move.customer_id1 == move.customer_id1 &&
                    tabu_move.vehicle1 == move.vehicle1 &&
                    tabu_move.vehicle2 == move.vehicle2) {
                    return true;
                }
            } else if (move.type == "2-0"){
                if (tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id2 == move.customer_id2
                    && tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2 ) {
                        return true;
                }
            } else if (move.type == "2-1"){
                if (((tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id2 == move.customer_id2 && tabu_move.customer_id3 == move.customer_id3) ||
                        (tabu_move.customer_id1 == move.customer_id3 && tabu_move.customer_id3 == move.customer_id1 && tabu_move.customer_id4 == move.customer_id2)) &&
                        ((tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2) ||
                        (tabu_move.vehicle1 == move.vehicle2 && tabu_move.vehicle2 == move.vehicle1))) {
                        return true;
                    }
            } else if (move.type == "2-2"){
                // Kiểm tra đơn giản hơn: chỉ cần khách hàng và xe giống nhau
                if (tabu_move.customer_id1 == move.customer_id1 && 
                    tabu_move.customer_id2 == move.customer_id2 &&
                    tabu_move.customer_id3 == move.customer_id3 &&
                    tabu_move.customer_id4 == move.customer_id4 &&
                    tabu_move.vehicle1 == move.vehicle1 && 
                    tabu_move.vehicle2 == move.vehicle2) {
                    return true;
                }
                // Kiểm tra move đảo ngược
                if (tabu_move.customer_id1 == move.customer_id3 && 
                    tabu_move.customer_id2 == move.customer_id4 &&
                    tabu_move.customer_id3 == move.customer_id1 &&
                    tabu_move.customer_id4 == move.customer_id2 &&
                    tabu_move.vehicle1 == move.vehicle2 && 
                    tabu_move.vehicle2 == move.vehicle1) {
                    return true;
                }
            } else if (move.type == "2-opt"){
                if (tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id3 == move.customer_id3
                    && tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2){
                        return true;
                    }
                if (tabu_move.customer_id1 == move.customer_id3 && tabu_move.customer_id3 == move.customer_id1
                    && tabu_move.vehicle1 == move.vehicle2 && tabu_move.vehicle2 == move.vehicle1) {
                        return true;
                }
            }
        }
    }
    return false;
}

Solution move_1_0(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    int cid = new_sol.route[v1][pos1];
    if (cid == depot_id) return current_sol; // không di chuyển depot
    int customer_count = 0;
    for (int node : new_sol.route[v1]) {
        if (node != depot_id) customer_count++;
    }
    
    if (customer_count <= 1) {
        // Xe chỉ còn 1 khách - không được di chuyển
        return current_sol;
    }
    if (pos2 == 0) {
        return current_sol;
    }
    
    // Kiểm tra không được chèn vào vị trí cuối cùng (sau depot kết thúc)
    if (pos2 == new_sol.route[v2].size()) {
        return current_sol;
    }
    
    // Kiểm tra không được di chuyển từ vị trí đầu hoặc cuối (depot)
    if (pos1 == 0 || pos1 == new_sol.route[v1].size() - 1) {
        return current_sol;
    }
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid);

    if (current_level != nullptr && is_merged_node(cid, *current_level)) {
        vector<int> group = get_merged_group(cid, *current_level);
        
        if (group.size() > 1) {
            // Tạo 2 solutions với 2 orientations khác nhau
            Solution sol_normal = new_sol;
            Solution sol_reversed = new_sol;
            
            // TÍNH COST CHO CẢ 2 ORIENTATIONS
            auto result_normal = find_best_orientation(sol_normal.route[v2], pos2, cid, current_level);
            
            // Tạo version đảo ngược bằng cách đảo thứ tự trong node_mapping
            vector<int> reversed_group = group;
            reverse(reversed_group.begin(), reversed_group.end());
            
            // Tính cost thủ công cho orientation đảo
            int prev_node = (pos2 > 0) ? sol_reversed.route[v2][pos2 - 1] : depot_id;
            int next_node = (pos2 < sol_reversed.route[v2].size() - 1) ? sol_reversed.route[v2][pos2 + 1] : depot_id;
            
            double cost_reversed = calculate_orientation_cost(prev_node, next_node, group, true, current_level);
            double cost_normal = calculate_orientation_cost(prev_node, next_node, group, false, current_level);
            
            // Chọn orientation tốt hơn
            if (cost_reversed < cost_normal - EPSILON) {
                // Đảo chiều tốt hơn - note lại để xử lý
                new_sol = sol_reversed;
                cout << "  → Move 1-0: Merged node " << cid << " REVERSED (cost " 
                     << cost_normal << " -> " << cost_reversed << ")" << endl;
            }
        }
    }

    evaluate_solution(new_sol, current_level);
    return new_sol;
}

Solution move_1_1(Solution current_sol, size_t v1, size_t node1, size_t v2, size_t node2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    int cid1 = new_sol.route[v1][node1];
    int cid2 = new_sol.route[v2][node2];
    if (cid1 == depot_id || cid2 == depot_id) return current_sol; // không di chuyển depot
    swap(new_sol.route[v1][node1], new_sol.route[v2][node2]);
    if (current_level != nullptr) {
        bool is_merged1 = is_merged_node(cid2, *current_level); // cid2 giờ ở vị trí node1
        bool is_merged2 = is_merged_node(cid1, *current_level); // cid1 giờ ở vị trí node2
        
        vector<Solution> candidates;
        candidates.push_back(new_sol); // Solution ban đầu
        
        //  NẾU CẢ 2 ĐỀU MERGED → THỬ 4 COMBINATIONS
        if (is_merged1 && is_merged2) {
            vector<int> group1 = get_merged_group(cid2, *current_level);
            vector<int> group2 = get_merged_group(cid1, *current_level);
            
            if (group1.size() > 1 && group2.size() > 1) {
                // Tính cost cho 4 combinations
                int prev1 = (node1 > 0) ? new_sol.route[v1][node1 - 1] : depot_id;
                int next1 = (node1 < new_sol.route[v1].size() - 1) ? new_sol.route[v1][node1 + 1] : depot_id;
                
                int prev2 = (node2 > 0) ? new_sol.route[v2][node2 - 1] : depot_id;
                int next2 = (node2 < new_sol.route[v2].size() - 1) ? new_sol.route[v2][node2 + 1] : depot_id;
                
                double cost_normal_normal = 
                    calculate_orientation_cost(prev1, next1, group1, false, current_level) +
                    calculate_orientation_cost(prev2, next2, group2, false, current_level);
                
                double cost_reversed_normal = 
                    calculate_orientation_cost(prev1, next1, group1, true, current_level) +
                    calculate_orientation_cost(prev2, next2, group2, false, current_level);
                
                double cost_normal_reversed = 
                    calculate_orientation_cost(prev1, next1, group1, false, current_level) +
                    calculate_orientation_cost(prev2, next2, group2, true, current_level);
                
                double cost_reversed_reversed = 
                    calculate_orientation_cost(prev1, next1, group1, true, current_level) +
                    calculate_orientation_cost(prev2, next2, group2, true, current_level);
                
                double min_cost = min({cost_normal_normal, cost_reversed_normal, 
                                      cost_normal_reversed, cost_reversed_reversed});
                
                if (abs(min_cost - cost_reversed_normal) < EPSILON) {
                    cout << "  → Move 1-1: cid1=" << cid1 << " normal, cid2=" << cid2 << " REVERSED" << endl;
                } else if (abs(min_cost - cost_normal_reversed) < EPSILON) {
                    cout << "  → Move 1-1: cid1=" << cid1 << " REVERSED, cid2=" << cid2 << " normal" << endl;
                } else if (abs(min_cost - cost_reversed_reversed) < EPSILON) {
                    cout << "  → Move 1-1: BOTH REVERSED" << endl;
                }
            }
        }
        //  NẾU CHỈ 1 MERGED
        else if (is_merged1) {
            find_best_orientation(new_sol.route[v1], node1, cid2, current_level);
        } else if (is_merged2) {
            find_best_orientation(new_sol.route[v2], node2, cid1, current_level);
        }
    }
    evaluate_solution(new_sol, current_level);
    return new_sol;
}

Solution move_2_0(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    int customer_count = 0;
    for (int node : new_sol.route[v1]) {
        if (node != depot_id) customer_count++;
    }
    
    if (customer_count <= 2) {
        // Xe chỉ còn 2 khách - không được di chuyển cả 2
        return current_sol;
    }
    int cid1 = new_sol.route[v1][pos1];
    int cid2 = new_sol.route[v1][pos1+1];
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid1);
    if (current_level != nullptr) {
        bool is_merged1 = is_merged_node(cid1, *current_level);
        bool is_merged2 = is_merged_node(cid2, *current_level);
        
        if (is_merged1) {
            find_best_orientation(new_sol.route[v2], pos2, cid1, current_level);
        }
        if (is_merged2) {
            find_best_orientation(new_sol.route[v2], pos2 + 1, cid2, current_level);
        }
    }
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2 + 1, cid2);
    evaluate_solution(new_sol, current_level);
    return new_sol;
}

Solution move_2_1(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    
    if (pos1 >= new_sol.route[v1].size() - 1 || pos2 >= new_sol.route[v2].size()) {
        return current_sol;
    }
    
    if (pos1 == 0 || pos2 == 0 || pos2 >= new_sol.route[v2].size() - 1) {
        return current_sol;
    }
    
    if (pos1 + 1 >= new_sol.route[v1].size() - 1) {
        return current_sol;
    }
    int customer_count_v1 = 0;
    for (int node : new_sol.route[v1]) {
        if (node != depot_id) customer_count_v1++;
    }
    
    if (customer_count_v1 <= 2) {
        // Xe chỉ còn 2 khách - swap sẽ tạo xe trống
        return current_sol;
    }
    
    int customer_count_v2 = 0;
    for (int node : new_sol.route[v2]) {
        if (node != depot_id) customer_count_v2++;
    }
    
    if (customer_count_v2 <= 1) {
        // Xe chỉ còn 1 khách - swap sẽ tạo xe trống
        return current_sol;
    }
    
    int cid1 = new_sol.route[v1][pos1];
    int cid2 = new_sol.route[v1][pos1+1];
    int cid3 = new_sol.route[v2][pos2];
    
    if (cid1 == depot_id || cid2 == depot_id || cid3 == depot_id) {
        return current_sol;
    }
    
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
    new_sol.route[v2].erase(new_sol.route[v2].begin() + pos2);
    new_sol.route[v1].insert(new_sol.route[v1].begin() + pos1, cid3);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid1);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2 + 1, cid2);

    if (current_level != nullptr) {
        bool is_merged1 = is_merged_node(cid1, *current_level);
        bool is_merged2 = is_merged_node(cid2, *current_level);
        bool is_merged3 = is_merged_node(cid3, *current_level);
        
        if (is_merged3) {
            vector<int> group3 = get_merged_group(cid3, *current_level);
            if (group3.size() > 1) {
                int prev1 = (pos1 > 0) ? new_sol.route[v1][pos1 - 1] : depot_id;
                int next1 = (pos1 < new_sol.route[v1].size() - 1) ? new_sol.route[v1][pos1 + 1] : depot_id;
                
                double cost_normal = calculate_orientation_cost(prev1, next1, group3, false, current_level);
                double cost_reversed = calculate_orientation_cost(prev1, next1, group3, true, current_level);
                
                if (cost_reversed < cost_normal - EPSILON) {
                    cout << "  → Move 2-1: cid3=" << cid3 << " REVERSED at v1" << endl;
                }
            }
        }
        
        if (is_merged1) {
            vector<int> group1 = get_merged_group(cid1, *current_level);
            if (group1.size() > 1) {
                int prev2 = (pos2 > 0) ? new_sol.route[v2][pos2 - 1] : depot_id;
                int next2 = (pos2 < new_sol.route[v2].size() - 1) ? new_sol.route[v2][pos2 + 1] : depot_id;
                
                double cost_normal = calculate_orientation_cost(prev2, next2, group1, false, current_level);
                double cost_reversed = calculate_orientation_cost(prev2, next2, group1, true, current_level);
                
                if (cost_reversed < cost_normal - EPSILON) {
                    cout << "  → Move 2-1: cid1=" << cid1 << " REVERSED at v2[" << pos2 << "]" << endl;
                }
            }
        }
        
        if (is_merged2) {
            vector<int> group2 = get_merged_group(cid2, *current_level);
            if (group2.size() > 1) {
                int prev2 = new_sol.route[v2][pos2];  // cid1
                int next2 = (pos2 + 2 < new_sol.route[v2].size()) ? new_sol.route[v2][pos2 + 2] : depot_id;
                
                double cost_normal = calculate_orientation_cost(prev2, next2, group2, false, current_level);
                double cost_reversed = calculate_orientation_cost(prev2, next2, group2, true, current_level);
                
                if (cost_reversed < cost_normal - EPSILON) {
                    cout << "  → Move 2-1: cid2=" << cid2 << " REVERSED at v2[" << (pos2+1) << "]" << endl;
                }
            }
        }
    }

    evaluate_solution(new_sol, current_level);
    return new_sol;
}

Solution move_2_2(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    
    if (pos1 >= new_sol.route[v1].size() - 1 || pos2 >= new_sol.route[v2].size() - 1) {
        return current_sol;
    }
    
    if (pos1 == 0 || pos2 == 0) {
        return current_sol;
    }
    
    if (pos1 + 1 >= new_sol.route[v1].size() - 1 || pos2 + 1 >= new_sol.route[v2].size() - 1) {
        return current_sol;
    }

    int customer_count_v1 = 0;
    for (int node : new_sol.route[v1]) {
        if (node != depot_id) customer_count_v1++;
    }
    
    int customer_count_v2 = 0;
    for (int node : new_sol.route[v2]) {
        if (node != depot_id) customer_count_v2++;
    }
    
    if (customer_count_v1 <= 2 || customer_count_v2 <= 2) {
        // Swap 2-2 sẽ tạo xe trống
        return current_sol;
    }
    
    int cid1 = new_sol.route[v1][pos1];
    int cid2 = new_sol.route[v1][pos1+1];
    int cid3 = new_sol.route[v2][pos2];
    int cid4 = new_sol.route[v2][pos2+1];
    
    if (cid1 == depot_id || cid2 == depot_id || cid3 == depot_id || cid4 == depot_id) {
        return current_sol;
    }
    
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
    new_sol.route[v2].erase(new_sol.route[v2].begin() + pos2 + 1);
    new_sol.route[v2].erase(new_sol.route[v2].begin() + pos2);
    new_sol.route[v1].insert(new_sol.route[v1].begin() + pos1, cid3);
    new_sol.route[v1].insert(new_sol.route[v1].begin() + pos1 + 1, cid4);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid1);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2 + 1, cid2);

    if (current_level != nullptr) {
        bool is_merged1 = is_merged_node(cid1, *current_level);
        bool is_merged2 = is_merged_node(cid2, *current_level);
        bool is_merged3 = is_merged_node(cid3, *current_level);
        bool is_merged4 = is_merged_node(cid4, *current_level);
        
        if (is_merged3 || is_merged4) {
            // cid3, cid4 giờ ở v1 tại pos1, pos1+1
            
            if (is_merged3) {
                vector<int> group3 = get_merged_group(cid3, *current_level);
                if (group3.size() > 1) {
                    int prev1 = (pos1 > 0) ? new_sol.route[v1][pos1 - 1] : depot_id;
                    int next1 = new_sol.route[v1][pos1 + 1];  // cid4
                    
                    double cost_normal = calculate_orientation_cost(prev1, next1, group3, false, current_level);
                    double cost_reversed = calculate_orientation_cost(prev1, next1, group3, true, current_level);
                    
                    if (cost_reversed < cost_normal - EPSILON) {
                        cout << "  → Move 2-2: cid3=" << cid3 << " REVERSED at v1[" << pos1 << "]" << endl;
                    }
                }
            }
            
            if (is_merged4) {
                vector<int> group4 = get_merged_group(cid4, *current_level);
                if (group4.size() > 1) {
                    int prev1 = new_sol.route[v1][pos1];  // cid3
                    int next1 = (pos1 + 2 < new_sol.route[v1].size()) ? new_sol.route[v1][pos1 + 2] : depot_id;
                    
                    double cost_normal = calculate_orientation_cost(prev1, next1, group4, false, current_level);
                    double cost_reversed = calculate_orientation_cost(prev1, next1, group4, true, current_level);
                    
                    if (cost_reversed < cost_normal - EPSILON) {
                        cout << "  → Move 2-2: cid4=" << cid4 << " REVERSED at v1[" << (pos1+1) << "]" << endl;
                    }
                }
            }
        }
        
        if (is_merged1 || is_merged2) {
            // cid1, cid2 giờ ở v2 tại pos2, pos2+1
            
            if (is_merged1) {
                vector<int> group1 = get_merged_group(cid1, *current_level);
                if (group1.size() > 1) {
                    int prev2 = (pos2 > 0) ? new_sol.route[v2][pos2 - 1] : depot_id;
                    int next2 = new_sol.route[v2][pos2 + 1];  // cid2
                    
                    double cost_normal = calculate_orientation_cost(prev2, next2, group1, false, current_level);
                    double cost_reversed = calculate_orientation_cost(prev2, next2, group1, true, current_level);
                    
                    if (cost_reversed < cost_normal - EPSILON) {
                        cout << "  → Move 2-2: cid1=" << cid1 << " REVERSED at v2[" << pos2 << "]" << endl;
                    }
                }
            }
            
            if (is_merged2) {
                vector<int> group2 = get_merged_group(cid2, *current_level);
                if (group2.size() > 1) {
                    int prev2 = new_sol.route[v2][pos2];  // cid1
                    int next2 = (pos2 + 2 < new_sol.route[v2].size()) ? new_sol.route[v2][pos2 + 2] : depot_id;
                    
                    double cost_normal = calculate_orientation_cost(prev2, next2, group2, false, current_level);
                    double cost_reversed = calculate_orientation_cost(prev2, next2, group2, true, current_level);
                    
                    if (cost_reversed < cost_normal - EPSILON) {
                        cout << "  → Move 2-2: cid2=" << cid2 << " REVERSED at v2[" << (pos2+1) << "]" << endl;
                    }
                }
            }
        }
    }

    evaluate_solution(new_sol, current_level);
    return new_sol;
}

Solution move_2opt(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    //  SAME TRIP
    if (v1 == v2) {
        if (contains_depot_in_range(new_sol.route[v1], pos1, pos2)) {
            return current_sol;
        }
        if (pos1 >= new_sol.route[v1].size() || pos2 >= new_sol.route[v1].size()) {
            return current_sol;
        }
        
        if (pos1 == 0 || pos2 >= new_sol.route[v1].size() - 1) {
            return current_sol;
        }
        
        if (pos1 >= pos2 || pos2 - pos1 < 2) {
            return current_sol;
        }
        
        // Lưu segment bị đảo
        vector<int> reversed_segment;
        for (size_t i = pos1; i <= pos2; i++) {
            reversed_segment.push_back(new_sol.route[v1][i]);
        }
        
        reverse(new_sol.route[v1].begin() + pos1, new_sol.route[v1].begin() + pos2 + 1);
        
        if (current_level != nullptr) {
            for (size_t i = pos1; i <= pos2; i++) {
                int node_id = new_sol.route[v1][i];
                if (node_id != depot_id && is_merged_node(node_id, *current_level)) {
                    vector<int> group = get_merged_group(node_id, *current_level);
                    if (group.size() > 1) {
                        int prev_node = (i > 0) ? new_sol.route[v1][i - 1] : depot_id;
                        int next_node = (i < new_sol.route[v1].size() - 1) ? new_sol.route[v1][i + 1] : depot_id;
                        
                        double cost_normal = calculate_orientation_cost(prev_node, next_node, group, false, current_level);
                        double cost_reversed = calculate_orientation_cost(prev_node, next_node, group, true, current_level);
                        
                        if (cost_reversed < cost_normal - EPSILON) {
                            cout << "  → Move 2-opt (intra): node=" << node_id 
                                 << " REVERSED at pos " << i << endl;
                        }
                    }
                }
            }
        }
    } 
    //  DIFFERENT TRIP
    else {
        if (contains_depot_in_range(new_sol.route[v1], pos1, new_sol.route[v1].size() - 2)) {
            return current_sol;
        }
        
        if (contains_depot_in_range(new_sol.route[v2], pos2, new_sol.route[v2].size() - 2)) {
            return current_sol;
        }
        if (pos1 >= new_sol.route[v1].size() - 1 || pos2 >= new_sol.route[v2].size() - 1) return current_sol;
        if (pos1 == 0 || pos2 == 0) return current_sol;

        if (pos1 >= new_sol.route[v1].size() || pos2 >= new_sol.route[v2].size()) {
            return current_sol;
        }
        
        vector<int> tail_v1(new_sol.route[v1].begin() + pos1, new_sol.route[v1].end() - 1);
        vector<int> tail_v2(new_sol.route[v2].begin() + pos2, new_sol.route[v2].end() - 1);
        
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1, new_sol.route[v1].end() - 1);
        new_sol.route[v2].erase(new_sol.route[v2].begin() + pos2, new_sol.route[v2].end() - 1);
        
        new_sol.route[v1].insert(new_sol.route[v1].end() - 1, tail_v2.begin(), tail_v2.end());
        new_sol.route[v2].insert(new_sol.route[v2].end() - 1, tail_v1.begin(), tail_v1.end());
        
        if (current_level != nullptr) {
            // Kiểm tra tail_v2 (giờ ở v1)
            size_t start_pos_v1 = pos1;
            for (size_t i = 0; i < tail_v2.size(); i++) {
                int node_id = tail_v2[i];
                if (node_id != depot_id && is_merged_node(node_id, *current_level)) {
                    vector<int> group = get_merged_group(node_id, *current_level);
                    if (group.size() > 1) {
                        size_t actual_pos = start_pos_v1 + i;
                        int prev_node = (actual_pos > 0) ? new_sol.route[v1][actual_pos - 1] : depot_id;
                        int next_node = (actual_pos < new_sol.route[v1].size() - 1) ? new_sol.route[v1][actual_pos + 1] : depot_id;
                        
                        double cost_normal = calculate_orientation_cost(prev_node, next_node, group, false, current_level);
                        double cost_reversed = calculate_orientation_cost(prev_node, next_node, group, true, current_level);
                        
                        if (cost_reversed < cost_normal - EPSILON) {
                            cout << "  → Move 2-opt (inter): node=" << node_id 
                                 << " REVERSED at v1[" << actual_pos << "]" << endl;
                        }
                    }
                }
            }
            
            // Kiểm tra tail_v1 (giờ ở v2)
            size_t start_pos_v2 = pos2;
            for (size_t i = 0; i < tail_v1.size(); i++) {
                int node_id = tail_v1[i];
                if (node_id != depot_id && is_merged_node(node_id, *current_level)) {
                    vector<int> group = get_merged_group(node_id, *current_level);
                    if (group.size() > 1) {
                        size_t actual_pos = start_pos_v2 + i;
                        int prev_node = (actual_pos > 0) ? new_sol.route[v2][actual_pos - 1] : depot_id;
                        int next_node = (actual_pos < new_sol.route[v2].size() - 1) ? new_sol.route[v2][actual_pos + 1] : depot_id;
                        
                        double cost_normal = calculate_orientation_cost(prev_node, next_node, group, false, current_level);
                        double cost_reversed = calculate_orientation_cost(prev_node, next_node, group, true, current_level);
                        
                        if (cost_reversed < cost_normal - EPSILON) {
                            cout << "  → Move 2-opt (inter): node=" << node_id 
                                 << " REVERSED at v2[" << actual_pos << "]" << endl;
                        }
                    }
                }
            }
        }
    }

    evaluate_solution(new_sol, current_level);
    return new_sol;
}

bool would_create_empty_vehicle(const Solution& sol, size_t vehicle_idx) {
    if (sol.route[vehicle_idx].size() <= 2) {
        for (int node : sol.route[vehicle_idx]) {
            if (node != depot_id) return false;
        }
        return true; // Xe trống
    }
    return false;
}

int count_customers_in_vehicle(const Solution& sol, size_t vehicle_idx) {
    int count = 0;
    for (int node : sol.route[vehicle_idx]) {
        if (node != depot_id) count++;
    }
    return count;
}

Solution tabu_search(Solution initial_sol, const LevelInfo &current_level, bool track_edge = true){
    update_node_index_cache(current_level);
    optimize_all_drone_routes(initial_sol, &current_level);
    remove_redundant_depots(initial_sol, &current_level);

    Solution best_sol = initial_sol;
    Solution current_sol = initial_sol;

    vector<TabuMove> tabu_list; // danh sách các move bị tabu
    int no_improve_count = 0;
    int last_depot_opt_iter = 0;
    int no_improve_segment_length = 0;
    const int max_no_improve_segment = 8;
    const int PHASE_1_END = MAX_ITER * 0.7;

    vector<string> move_types = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
    
    for (int iter = 0; iter < MAX_ITER && no_improve_count < MAX_NO_IMPROVE; iter++){
        bool in_phase_1 = (iter < PHASE_1_END);
        bool should_depot_opt = false;
        if (in_phase_1) {
            if (iter == 10 || iter == 30 || iter == 60) {
                should_depot_opt = true;
            } else if (current_sol.drone_violation > 10.0 || current_sol.waiting_violation > 20.0) {
                if (iter - last_depot_opt_iter >= 200) {
                    should_depot_opt = true;
                }
            } else if (no_improve_count >= 100 && (iter - last_depot_opt_iter) >= 50) {
                should_depot_opt = true;
            }
        } else {
            if (iter % 50 == 0) {
                if (current_sol.drone_violation > 0.5 || current_sol.waiting_violation > 0.5) {
                    should_depot_opt = true;
                }
            } 
            else if (no_improve_count >= 30 && (iter - last_depot_opt_iter) >= 30) {
                should_depot_opt = true;
            } 
            else if (iter > MAX_ITER - 100 && (iter - last_depot_opt_iter) >= 30) {
                if (current_sol.drone_violation > EPSILON || current_sol.waiting_violation > EPSILON) {
                    should_depot_opt = true;
                }
            } 
        }

        if (should_depot_opt) {
            Solution temp_sol = current_sol;
            double old_fitness = temp_sol.fitness;
            
            optimize_all_drone_routes(temp_sol, &current_level);
            remove_redundant_depots(temp_sol, &current_level);
            
            if (temp_sol.fitness < current_sol.fitness - EPSILON) {
                current_sol = temp_sol;
                last_depot_opt_iter = iter;
                                
                if (current_sol.fitness < best_sol.fitness - EPSILON) {
                    best_sol = current_sol;
                    no_improve_count = 0;

                    if (track_edge) {
                        update_edge_frequency(best_sol);
                    }
                }
            }
        }

        auto is_merged_group = [&](int node_id) -> bool {
            auto it = current_level.node_mapping.find(node_id);
            return (it != current_level.node_mapping.end() && it->second.size() > 1);
        };

        double best_Neighbor_fitness = DBL_MAX;
        Solution best_Neighbor_sol = current_sol;
        double current_fitness = current_sol.fitness;
        TabuMove best_move;
        int best_move_node1 = -1, best_move_node2 = -1, best_move_node3 = -1, best_move_node4 = -1;
        bool improved = false;
        //string move_type = "2-2";

        int move_type_idx = select_move_type();
        //int move_type_idx = rand() % MOVE_SET.size();
        string move_type = MOVE_SET[move_type_idx];
        used_count[move_type_idx]++;
        bool segment_improved = false;

        // move 1-0
        if (move_type == "1-0") {
            for (size_t v1 = 0; v1 < current_sol.route.size(); v1++) {
                for (size_t pos1 = 1; pos1 < current_sol.route[v1].size()-1; pos1++) {
                    int n1 = current_sol.route[v1][pos1];
                    if (n1 == depot_id) continue;

                    int customer_count_v1 = count_customers_in_vehicle(current_sol, v1);
                    if (customer_count_v1 <= 1) {
                        continue; 
                    }

                    for (size_t v2 = 0; v2 < current_sol.route.size(); v2++) {
                        if (v1 == v2) continue;
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++) {
                            if (get_type(n1, &current_level) == 1 && (vehicles[v2].is_drone || vehicles[v1].is_drone)) continue; // C1 không thể giao cho drone

                            Solution new_sol = move_1_0(current_sol, v1, pos1, v2, pos2, &current_level);
                            TabuMove move = {"1-0", n1, -1, -1, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);

                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = -1;
                                best_move_node3 = -1;
                                best_move_node4 = -1;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = new_sol.fitness;
                                    best_Neighbor_sol = new_sol;
                                    best_move = move;
                                    best_move_node1 = n1;   
                                    best_move_node2 = -1;
                                    best_move_node3 = -1;
                                    best_move_node4 = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // move 1-1
        if (move_type == "1-1") {
            for (size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for (size_t pos1 = 1; pos1 < current_sol.route[v1].size() -1 ; pos1++) {
                    int n1 = current_sol.route[v1][pos1];
                    if (n1 == depot_id) continue;
                    for (size_t v2 = 0; v2 < vehicles.size(); v2++) {
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++) {
                            int n2 = current_sol.route[v2][pos2];
                            if (n2 == depot_id || n1 == n2 || get_type(n1, &current_level) != get_type(n2, &current_level) || ((abs(int(pos1)-int(pos2)) <= 1) && (v1 == v2))) continue;

                            Solution new_sol = move_1_1(current_sol, v1, pos1, v2, pos2, &current_level);
                            TabuMove move = {"1-1", n1, -1, n2, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);

                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = -1;
                                best_move_node3 = n2;
                                best_move_node4 = -1;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = new_sol.fitness;
                                    best_Neighbor_sol = new_sol;
                                    best_move = move;
                                    best_move_node1 = n1;
                                    best_move_node2 = -1;
                                    best_move_node3 = n2;
                                    best_move_node4 = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (move_type == "2-0") {
            for(size_t v1 = 0; v1 < vehicles.size(); v1++){
                for(size_t pos1 = 1; pos1 < current_sol.route[v1].size()-2; pos1++){
                    int n1 = current_sol.route[v1][pos1];
                    int n2 = current_sol.route[v1][pos1+1];
                    if (n1 == depot_id || n2 == depot_id) continue;
                    for (size_t v2 = 0; v2 < vehicles.size(); v2++){
                        if (v1 == v2) continue;
                        if ((get_type(n1, &current_level) == 1 || get_type(n2, &current_level) == 1) && vehicles[v2].is_drone) continue;
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++){

                            Solution new_sol = move_2_0(current_sol, v1, pos1, v2, pos2, &current_level);
                            TabuMove move = {"2-0", n1, n2, -1, -1, (int)v1, (int)v2, (int)pos1, (int)pos1+1, (int)pos2, (int)pos2+1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);

                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = n2;
                                best_move_node3 = -1;
                                best_move_node4 = -1;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = new_sol.fitness;
                                    best_Neighbor_sol = new_sol;
                                    best_move = move;
                                    best_move_node1 = n1;
                                    best_move_node2 = n2;
                                    best_move_node3 = -1;
                                    best_move_node4 = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // move 2-1
        if (move_type == "2-1") {
            for(size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for(size_t pos1 = 1; pos1 < current_sol.route[v1].size() - 2; pos1++) {
                    int n1 = current_sol.route[v1][pos1];
                    int n2 = current_sol.route[v1][pos1+1];
                    if (n1 == depot_id || n2 == depot_id) continue;
                    for (size_t v2 = 0; v2 < vehicles.size(); v2++){
                        if (v1 == v2) continue;
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++){
                            int n3 = current_sol.route[v2][pos2];
                            if (n3 == depot_id) continue;
                            if (v1 == v2 && (abs(int(pos1)-int(pos2)) <= 2)) continue;
                            if ((get_type(n1, &current_level) == 1 || get_type(n2, &current_level) == 1) && vehicles[v2].is_drone) continue;
                            if (get_type(n3, &current_level) == 1 && vehicles[v1].is_drone) continue;
                            Solution new_sol = move_2_1(current_sol, v1, pos1, v2, pos2, &current_level);
                            TabuMove move = {"2-1", n1, n2, n3, -1, (int)v1, (int)v2, (int)pos1, (int)pos1+1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);
                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = n2;
                                best_move_node3 = n3;
                                best_move_node4 = -1;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = new_sol.fitness;
                                    best_Neighbor_sol = new_sol;
                                    best_move = move;
                                    best_move_node1 = n1;
                                    best_move_node2 = n2;
                                    best_move_node3 = n3;
                                    best_move_node4 = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (move_type == "2-2"){
            for (size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for (size_t pos1 = 1; pos1 < current_sol.route[v1].size() -2; pos1++){
                    int n1 = current_sol.route[v1][pos1];
                    int n2 = current_sol.route[v1][pos1+1];
                    if (n1 == depot_id || n2 == depot_id) continue;
                    for (size_t v2 = 0; v2 < vehicles.size(); v2++){
                        if (v1 == v2) continue;
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size() - 2; pos2++){
                            int n3 = current_sol.route[v2][pos2];
                            int n4 = current_sol.route[v2][pos2+1];
                            if (n3 == depot_id || n4 == depot_id) continue;
                            if ((get_type(n1, &current_level) == 1 || get_type(n2, &current_level) == 1) && vehicles[v2].is_drone) continue;
                            if ((get_type(n3, &current_level) == 1 || get_type(n4, &current_level) == 1) && vehicles[v1].is_drone) continue;

                            Solution new_sol = move_2_2(current_sol, v1, pos1, v2, pos2, &current_level);
                            TabuMove move = {"2-2", n1, n2, n3, n4, int(v1), int(v2), int(pos1), int(pos1+1), int(pos2), int(pos2+1), TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);
                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = n2;
                                best_move_node3 = n3;
                                best_move_node4 = n4;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = new_sol.fitness;
                                    best_Neighbor_sol = new_sol;
                                    best_move = move;
                                    best_move_node1 = n1;
                                    best_move_node2 = n2;
                                    best_move_node3 = n3;
                                    best_move_node4 = n4;
                                }
                            }
                        }
                    }
                }
            }
        }

        // move 2-opt
        if (move_type == "2-opt") {
            // Intra-route 2-opt (cùng xe)
            for(size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for(size_t pos1 = 1; pos1 < current_sol.route[v1].size() - 1; pos1++) {
                    if (current_sol.route[v1][pos1] == depot_id) continue;
                    for(size_t pos2 = pos1 + 2; pos2 < current_sol.route[v1].size() - 1; pos2++) {
                        if (current_sol.route[v1][pos2] == depot_id) continue;

                        int customer_at_pos1 = current_sol.route[v1][pos1];
                        int customer_at_pos2 = current_sol.route[v1][pos2];

                        Solution new_sol = move_2opt(current_sol, v1, pos1, v1, pos2, &current_level); // Cùng xe v1
                        TabuMove move = {"2-opt", customer_at_pos1, -1, customer_at_pos2, -1, (int)v1, (int)v1, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
                        bool tabu = is_tabu(tabu_list, move);
                        
                        if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                            best_Neighbor_fitness = new_sol.fitness;
                            best_Neighbor_sol = new_sol;
                            best_move = move;
                            best_move_node1 = customer_at_pos1;
                            best_move_node2 = -1;
                            best_move_node3 = customer_at_pos2;
                            best_move_node4 = -1;
                            improved = true;
                        } else if (improved == false) {
                            if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = customer_at_pos1;
                                best_move_node2 = -1;
                                best_move_node3 = customer_at_pos2;
                                best_move_node4 = -1;
                            }
                        }
                    }
                }
            }
            
            // Inter-route 2-opt (khác xe)
            for(size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for(size_t v2 = v1 + 1; v2 < vehicles.size(); v2++) {
                    for(size_t pos1 = 1; pos1 < current_sol.route[v1].size() - 1; pos1++) {
                        if (current_sol.route[v1][pos1] == depot_id) continue;
                        for(size_t pos2 = 1; pos2 < current_sol.route[v2].size() - 1; pos2++) {
                            if (current_sol.route[v2][pos2] == depot_id) continue;
                            bool invalid_move = false;
                            
                            if (vehicles[v2].is_drone) {
                                for (size_t i = pos1; i < current_sol.route[v1].size() - 1; i++) {
                                    int cid = current_sol.route[v1][i];
                                    if (cid != depot_id && get_type(cid, &current_level) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (!invalid_move && vehicles[v1].is_drone) {
                                for (size_t i = pos2; i < current_sol.route[v2].size() - 1; i++) {
                                    int cid = current_sol.route[v2][i];
                                    if (cid != depot_id && get_type(cid, &current_level) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (invalid_move) continue;

                            int customer_at_pos1 = current_sol.route[v1][pos1];
                            int customer_at_pos2 = current_sol.route[v2][pos2];
                            
                            Solution new_sol = move_2opt(current_sol, v1, pos1, v2, pos2, &current_level); // Khác xe v1 và v2
                            TabuMove move = {"2-opt", customer_at_pos1, -1, customer_at_pos2, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);
                            
                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_move = move;
                                best_move_node1 = customer_at_pos1;
                                best_move_node2 = -1;
                                best_move_node3 = customer_at_pos2;
                                best_move_node4 = -1;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (new_sol.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = new_sol.fitness;
                                    best_Neighbor_sol = new_sol;
                                    best_move = move;
                                    best_move_node1 = customer_at_pos1;
                                    best_move_node2 = -1;
                                    best_move_node3 = customer_at_pos2;
                                    best_move_node4 = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        bool should_apply_move = false;

        if (move_type == "1-0") {
            should_apply_move = (best_move_node1 != -1);
        }
        else if (move_type == "1-1") {
            should_apply_move = (best_move_node1 != -1 && best_move_node3 != -1);
        }
        else if (move_type == "2-0") {
            should_apply_move = (best_move_node1 != -1 && best_move_node2 != -1);
        }
        else if (move_type == "2-1") {
            should_apply_move = (best_move_node1 != -1 && best_move_node2 != -1 && best_move_node3 != -1);
        }
        else if (move_type == "2-2") {
            should_apply_move = (best_move_node1 != -1 && best_move_node2 != -1 && best_move_node3 != -1 && best_move_node4 != -1);
        }
        else if (move_type == "2-opt") {
            should_apply_move = (best_move_node1 != -1 && best_move_node3 != -1);
        }
        
        if (should_apply_move) {
            current_sol = best_Neighbor_sol;
            evaluate_solution(current_sol, &current_level);

            /*cout << "Iter: " << iter << " Move: " << move_type 
                 << " current makespan: " << current_sol.makespan 
                 << ", drone_violation: " << current_sol.drone_violation 
                 << ", waiting_violation: " << current_sol.waiting_violation 
                 << ", fitness: " << current_sol.fitness << endl;
            cout << "Route details:" << endl;
            for (size_t v = 0; v < current_sol.route.size(); v++) {
                cout << "Vehicle " << v << ": ";
                for (int cid : current_sol.route[v]) cout << cid << " ";
                cout << endl;
            }*/

            // Cập nhật tabu list
            for (auto it = tabu_list.begin(); it != tabu_list.end(); ) {
                it->tenure--;
                if (it->tenure <= 0) {
                    it = tabu_list.erase(it);
                } else {
                    ++it;
                }
            }
            tabu_list.push_back(best_move);
            /*cout << "Tabu move added: type=" << best_move.type
                 << ", customer1=" << best_move.customer_id1
                 << ", customer2=" << best_move.customer_id2
                 << ", customer3=" << best_move.customer_id3
                 << ", customer4=" << best_move.customer_id4
                 << ", vehicle1=" << best_move.vehicle1
                 << ", vehicle2=" << best_move.vehicle2
                 << ", pos1=" << best_move.pos1
                 << ", pos2=" << best_move.pos2
                 << ", pos3=" << best_move.pos3
                 << ", pos4=" << best_move.pos4
                 << ", tenure=" << best_move.tenure << endl;*/
            TabuMove reverse_move;
            if (move_type == "1-0") {
                reverse_move = {"1-0", best_move_node1, -1, -1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, -1, TABU_TENURE};
            }
            else if (move_type == "1-1") {
                reverse_move = {"1-1", best_move_node3, -1, best_move_node1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, -1, TABU_TENURE};
            }
            else if (move_type == "2-0") {
                reverse_move = {"2-0", best_move_node1, best_move_node2, -1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, best_move.pos4, best_move.pos1, best_move.pos2, TABU_TENURE};
            }
            else if (move_type == "2-1") {
                reverse_move = {"2-1", best_move_node3, -1, best_move_node1, best_move_node2, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, best_move.pos2, TABU_TENURE};
            }
            else if (move_type == "2-2") {
                reverse_move = {"2-2", best_move_node3, best_move_node4, best_move_node1, best_move_node2, best_move.vehicle2, best_move.vehicle1, best_move.pos3, best_move.pos4, best_move.pos1, best_move.pos2, TABU_TENURE};
            }
            else if (move_type == "2-opt") {
                reverse_move = {"2-opt", best_move_node3, -1, best_move_node1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, -1, TABU_TENURE};
            }
            tabu_list.push_back(reverse_move);

            /*cout << "Tabu move added: type=" << reverse_move.type
                 << ", customer1=" << reverse_move.customer_id1
                 << ", customer2=" << reverse_move.customer_id2
                 << ", customer3=" << reverse_move.customer_id3
                 << ", customer4=" << reverse_move.customer_id4
                 << ", vehicle1=" << reverse_move.vehicle1
                 << ", vehicle2=" << reverse_move.vehicle2
                 << ", pos1=" << reverse_move.pos1
                 << ", pos2=" << reverse_move.pos2
                 << ", pos3=" << reverse_move.pos3
                 << ", pos4=" << reverse_move.pos4
                 << ", tenure=" << reverse_move.tenure << endl;*/
            
            if (current_sol.is_feasible && current_sol.fitness < best_sol.fitness - EPSILON){
                best_sol = current_sol;
                no_improve_count = 0;
                scorePi[move_type_idx] += delta1;
                segment_improved = true;
                if (track_edge) {
                    update_edge_frequency(best_sol);
                }
            } else if (current_sol.fitness < current_fitness - EPSILON) {
                scorePi[move_type_idx] += delta2;
                no_improve_count++;
            } else {
                scorePi[move_type_idx] += delta3;
                no_improve_count++;
            }
        } else no_improve_count++;

        if ((iter + 1)% SEGMENT_LENGTH == 0) {
            update_weights();
            if (segment_improved) {
                no_improve_segment_length = 0;
            } else {
                no_improve_segment_length++;
            }
            /*cout << "SEGMENT " << (iter + 1)/SEGMENT_LENGTH << " COMPLETE" << endl;
            cout << "No improve segments: " << no_improve_segment_length <<"/"<< max_no_improve_segment << endl;
            cout << "Updated weights: ";
            for (size_t i = 0; i < MOVE_SET.size(); i++) {
                cout << MOVE_SET[i] << "=" << weights[i] << " ";
            }
            cout << endl;
            cout << "Current best fitness: " << best_sol.fitness << endl;*/
        }
    }
    optimize_all_drone_routes(best_sol, &current_level);
    remove_redundant_depots(best_sol, &current_level);
    return best_sol;
}

void create_coarse_distance_matrix(LevelInfo& next_level, const LevelInfo& current_level,const vector<vector<double>>& curr_distances,const vector<vector<double>>& curr_original_distances,
                                   vector<vector<double>>& next_distances,vector<vector<double>>& next_original_distances){
    int n = next_level.nodes.size();
    next_distances.resize(n, vector<double>(n, 0.0));
    next_original_distances.resize(n, vector<double>(n, 0.0));

    map<int, int> original_to_current_node;
    for (const auto& pair : current_level.node_mapping) {
        int current_node_id = pair.first;
        const vector<int>& original_nodes = pair.second;
        
        for (int orig : original_nodes) {
            original_to_current_node[orig] = current_node_id;
        }
    }

    /*cout << "\n=== REVERSE MAPPING (Original → Current Level) ===" << endl;
    for (const auto& pair : original_to_current_node) {
        cout << "Original " << pair.first << " → Current level node " << pair.second << endl;
    }*/

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (i == j){
                next_distances[i][j] = 0.0;
                next_original_distances[i][j] = 0.0;
                continue;
            }

            vector<int> next_group_i = next_level.node_mapping[next_level.nodes[i].id];
            vector<int> next_group_j = next_level.node_mapping[next_level.nodes[j].id];

            /*cout << "\nCalculating distance [" << next_level.nodes[i].id << "][" << next_level.nodes[j].id << "]" << endl;
            cout << "  Next group i: [";
            for (int x : next_group_i) cout << x << " ";
            cout << "]" << endl;
            cout << "  Next group j: [";
            for (int x : next_group_j) cout << x << " ";
            cout << "]" << endl;*/

            vector<int> current_group_i, current_group_j;
            
            for (int orig : next_group_i) {
                auto it = original_to_current_node.find(orig);
                if (it != original_to_current_node.end()) {
                    int current_node = it->second;
                    if (find(current_group_i.begin(), current_group_i.end(), current_node) == current_group_i.end()) {
                        current_group_i.push_back(current_node);
                    }
                }
            }
            
            for (int orig : next_group_j) {
                auto it = original_to_current_node.find(orig);
                if (it != original_to_current_node.end()) {
                    int current_node = it->second;
                    if (find(current_group_j.begin(), current_group_j.end(), current_node) == current_group_j.end()) {
                        current_group_j.push_back(current_node);
                    }
                }
            }

            /*cout << "  Current level group i: [";
            for (int x : current_group_i) cout << x << " ";
            cout << "]" << endl;
            cout << "  Current level group j: [";
            for (int x : current_group_j) cout << x << " ";
            cout << "]" << endl;*/

            if (current_group_i.size() == 1 && current_group_j.size() == 1){
                int curr_i = current_group_i[0];
                int curr_j = current_group_j[0];

                int idx_i = find_node_index_fast(curr_i);
                int idx_j = find_node_index_fast(curr_j);

                if (idx_i != -1 && idx_j != -1){
                    next_distances[i][j] = curr_distances[idx_i][idx_j];
                    next_original_distances[i][j] = curr_original_distances[idx_i][idx_j];
                    
                    //cout << "  → Single to single: distance=" << next_distances[i][j] << endl;
                }
            } else {
                int exit_current_i = current_group_i.back();
                int idx_exit_i = find_node_index_fast(exit_current_i);
                
                int entry_current_j = current_group_j.front();
                int idx_entry_j = find_node_index_fast(entry_current_j);

                double total_distance = 0.0;
                
                if (current_group_i.size() > 1) {
                    //cout << "  → Group i has " << current_group_i.size() << " nodes in current level" << endl;
                    for (size_t k = 0; k < current_group_i.size() - 1; k++) {
                        int from = current_group_i[k];
                        int to = current_group_i[k + 1];
                        int idx_from = find_node_index_fast(from);
                        int idx_to = find_node_index_fast(to);

                        if (idx_from != -1 && idx_to != -1) {
                            double d = curr_distances[idx_from][idx_to];
                            total_distance += d;
                            //cout << "    " << from << " → " << to << ": " << d << endl;
                        }
                    }
                }

                if (idx_exit_i != -1 && idx_entry_j != -1) {
                    double d = curr_distances[idx_exit_i][idx_entry_j];
                    total_distance += d;
                    //cout << "  → Between groups: " << exit_current_i << " → " << entry_current_j << ": " << d << endl;
                }

                if (current_group_j.size() > 1) {
                    //cout << "  → Group j has " << current_group_j.size() << " nodes in current level" << endl;
                    for (size_t k = 0; k < current_group_j.size() - 1; k++) {
                        int from = current_group_j[k];
                        int to = current_group_j[k + 1];
                        int idx_from = find_node_index_fast(from);
                        int idx_to = find_node_index_fast(to);

                        if (idx_from != -1 && idx_to != -1) {
                            double d = curr_distances[idx_from][idx_to];
                            total_distance += d;
                            //cout << "    " << from << " → " << to << ": " << d << endl;
                        }
                    }
                }

                next_distances[i][j] = total_distance;
                //cout << "  → Total distance: " << total_distance << endl;

                if (idx_exit_i != -1 && idx_entry_j != -1){
                    next_original_distances[i][j] = curr_original_distances[idx_exit_i][idx_entry_j];
                    //cout << "  → Original distance: " << next_original_distances[i][j] << endl;
                }
            }
        }
    }
}

void classify_customers(LevelInfo& level){
    level.C1_level.clear();
    level.C2_level.clear();
    for (const auto& node : level.nodes){
        if (node.id == depot_id) continue;
        if (node.c1_or_c2 == 0){
            level.C1_level.push_back(node);
        } else {
            level.C2_level.push_back(node);
        }
    }
    level.num_customers = level.C1_level.size() + level.C2_level.size();
}

vector<tuple<int, int, int>> collect_merge_candidates(const LevelInfo& current_level, const Solution& best_solution){
    vector<tuple<int, int, int>> candidates;
    map<pair<int,int>, int> solution_edges;
    
    for (size_t v = 0; v < best_solution.route.size(); v++) {
        
        const vector<int>& route = best_solution.route[v];
        for (size_t i = 0; i < route.size() - 1; i++) {
            int from_node = route[i];
            int to_node = route[i + 1];
            
            if (from_node == depot_id || to_node == depot_id) continue;
            
            pair<int,int> edge = make_pair(from_node, to_node);
            solution_edges[edge]++;
        }
    }

    for (const auto& edge_pair : solution_edges){
        int from_node = edge_pair.first.first;
        int to_node = edge_pair.first.second;

        if (from_node == depot_id || to_node == depot_id) continue;

        int count = edge_pair.second; 
        
        int frequency = 0;
        auto it = edge_frequency.find(edge_pair.first);
        if (it != edge_frequency.end()) {
            frequency = it->second;
        }
        
        // Kiểm tra xem node có tồn tại trong level hiện tại không
        int idx_from = find_node_index_fast(from_node);
        int idx_to = find_node_index_fast(to_node);
        if (idx_from == -1 || idx_to == -1) {
            /*cout << "  Skipping edge (" << from_node << ", " << to_node 
                 << ") - node not found in current level" << endl;*/
            continue;
        }

        const Node& node_from = current_level.nodes[idx_from];
        const Node& node_to = current_level.nodes[idx_to];
        
        bool same_type = (node_from.c1_or_c2 == 0 && node_to.c1_or_c2 == 0) || 
                        (node_from.c1_or_c2 > 0 && node_to.c1_or_c2 > 0);
        
        if (same_type){
            candidates.emplace_back(make_tuple(frequency, from_node, to_node));
            cout << "Candidate edge: (" << from_node << ", " << to_node 
                 << ") frequency=" << frequency  << endl;
        }
    }

    sort(candidates.begin(), candidates.end(), 
         [](const tuple<int, int, int>& a, const tuple<int, int, int>& b) {
        return get<0>(a) > get<0>(b);
    });
    
    return candidates;
}

LevelInfo merge_customers(const LevelInfo& current_level, const Solution& best_solution, 
                         const vector<vector<double>>& curr_distances,
                         const vector<vector<double>>& curr_original_distances) {
    LevelInfo next_level;
    next_level.level_id = current_level.level_id + 1;
    
    vector<tuple<int,int,int>> candidates = collect_merge_candidates(current_level, best_solution);
    
    // Tính 20% số CẠNH, không phải nodes
    int num_to_merge = max(1, (int)(candidates.size() * 0.2));
    
    cout << "\n=== MERGING " << num_to_merge << " / " << candidates.size() 
         << " EDGES (20%) ===" << endl;
    
    set<int> merged_nodes;
    vector<vector<int>> merged_groups;
    
    for (int i = 0; i < num_to_merge && i < candidates.size(); i++) {
        int frequency = get<0>(candidates[i]);
        int node_a = get<1>(candidates[i]);
        int node_b = get<2>(candidates[i]);

        if (node_a == depot_id || node_b == depot_id) {
            continue;
        }
        
        bool already_merged_together = false;
        for (const auto& group : merged_groups) {
            bool has_a = (find(group.begin(), group.end(), node_a) != group.end());
            bool has_b = (find(group.begin(), group.end(), node_b) != group.end());
            if (has_a && has_b) {
                already_merged_together = true;
                break;
            }
        }
        
        if (already_merged_together) {
            continue;
        }
        
        // Tìm hoặc tạo group chứa node_a và node_b
        int group_idx_a = -1, group_idx_b = -1;
        
        for (size_t g = 0; g < merged_groups.size(); g++) {
            if (find(merged_groups[g].begin(), merged_groups[g].end(), node_a) != merged_groups[g].end()) {
                group_idx_a = g;
            }
            if (find(merged_groups[g].begin(), merged_groups[g].end(), node_b) != merged_groups[g].end()) {
                group_idx_b = g;
            }
        }
        
        // Case 1: Cả 2 đều chưa có trong group nào -> Tạo group mới
        if (group_idx_a == -1 && group_idx_b == -1) {
            merged_groups.push_back({node_a, node_b});
            merged_nodes.insert(node_a);
            merged_nodes.insert(node_b);
            cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b << ") freq=" << frequency << " → NEW GROUP" << endl;
        }
        // Case 2: node_a đã có group, node_b chưa -> Thêm node_b vào group của node_a
        else if (group_idx_a != -1 && group_idx_b == -1) {
            // Kiểm tra node a ở đầu hay cuối group
            if (merged_groups[group_idx_a].back() == node_a){
                merged_groups[group_idx_a].push_back(node_b);
            } else if (merged_groups[group_idx_a].front() == node_a){
                merged_groups[group_idx_a].insert(merged_groups[group_idx_a].begin(), node_b);
            } else {
                // Không nên xảy ra
                cout << "⚠️  Warning: node " << node_a << " not at group ends!" << endl;
                continue;
            }
            merged_nodes.insert(node_b);
            cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b 
                 << ") freq=" << frequency << " → ADD TO GROUP " << group_idx_a << endl;
        }
        // Case 3: node_b đã có group, node_a chưa -> Thêm node_a vào group của node_b
        else if (group_idx_a == -1 && group_idx_b != -1) {
            if (merged_groups[group_idx_b].front() == node_b){
                merged_groups[group_idx_b].insert(merged_groups[group_idx_b].begin(), node_a);
            } else if (merged_groups[group_idx_b].back() == node_b){
                merged_groups[group_idx_b].push_back(node_a);
            } else {
                // Không nên xảy ra
                cout << "⚠️  Warning: node " << node_b << " not at group ends!" << endl;
                continue;
            }
            merged_nodes.insert(node_a);
            cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b 
                 << ") freq=" << frequency << " → ADD TO GROUP " << group_idx_b << endl;
        }
        // Case 4: Cả 2 đã có group khác nhau → Merge 2 groups
        else if (group_idx_a != group_idx_b) {
            // Chỉ nối nếu node_a ở cuối group_a VÀ node_b ở đầu group_b
            if (merged_groups[group_idx_a].back() == node_a && 
                merged_groups[group_idx_b].front() == node_b) {
                // Nối group_b vào cuối group_a
                merged_groups[group_idx_a].insert(
                    merged_groups[group_idx_a].end(),
                    merged_groups[group_idx_b].begin(),
                    merged_groups[group_idx_b].end()
                );
                merged_groups.erase(merged_groups.begin() + group_idx_b);
                cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b 
                     << ") freq=" << frequency << " → CONNECT GROUPS" << endl;
            } else {
                cout << "  ⚠️  Cannot connect - nodes not at boundaries" << endl;
            }
        }
    }
    
    cout << "\n=== FINAL MERGED GROUPS ===" << endl;
    for (size_t i = 0; i < merged_groups.size(); i++) {
        cout << "Group " << (i+1) << ": [";
        for (size_t j = 0; j < merged_groups[i].size(); j++) {
            cout << merged_groups[i][j];
            if (j < merged_groups[i].size() - 1) cout << " -> ";
        }
        cout << "]" << endl;
    }
    
    if (merged_groups.empty()) {
        cout << "⚠️  No groups formed! Returning current level." << endl;
        return current_level;
    }
    
    // Đặt tên mới cho node
    int next_node_id = (next_level.level_id) * 1000;
    
    next_level.nodes.push_back({depot_id, 0.0, 0.0, -1.0, DBL_MAX});
    next_level.node_mapping[depot_id] = {depot_id};
    
    for (const auto& group : merged_groups) {
        int first_node_id = group[0];
        int idx = find_node_index_fast(first_node_id);
        
        if (idx != -1) {
            const Node& first_node = current_level.nodes[idx];
            Node merged_node = {
                next_node_id++,
                first_node.x,
                first_node.y,
                first_node.c1_or_c2,
                first_node.limit_wait
            };
            next_level.nodes.push_back(merged_node);
            
            vector<int> original_nodes;
            for (int node_id : group) {
                auto it = current_level.node_mapping.find(node_id);
                if (it != current_level.node_mapping.end()) {
                    for (int orig : it->second) {
                        if (find(original_nodes.begin(), original_nodes.end(), orig) 
                            == original_nodes.end()) {
                            original_nodes.push_back(orig);
                        }
                    }
                } else {
                    if (find(original_nodes.begin(), original_nodes.end(), node_id) 
                        == original_nodes.end()) {
                        original_nodes.push_back(node_id);
                    }
                }
            }
            
            next_level.node_mapping[merged_node.id] = original_nodes;
        }
    }
    
    // thêm những node không bị merge vào next level
    for (const auto& node : current_level.nodes) {
        if (node.id == depot_id) continue;
        if (merged_nodes.find(node.id) == merged_nodes.end()) {
            next_level.nodes.push_back(node);
            auto it = current_level.node_mapping.find(node.id);
            if (it != current_level.node_mapping.end()) {
                next_level.node_mapping[node.id] = it->second;
            } else {
                next_level.node_mapping[node.id] = {node.id};
            }
        }
    }
    
    classify_customers(next_level);
    
    cout << "\n✅ Level " << next_level.level_id << " created: " 
         << current_level.nodes.size() << " → " << next_level.nodes.size() 
         << " nodes (merged " << (current_level.nodes.size() - next_level.nodes.size()) 
         << ")" << endl;
    
    return next_level;
}

Solution project_solution_to_next_level(const Solution& old_sol, const LevelInfo& old_level, const LevelInfo& next_level){
    Solution new_sol;
    new_sol.route.resize(old_sol.route.size());
    
    map<int, int> old_to_new_mapping;
    
    old_to_new_mapping[depot_id] = depot_id;
    
    for (const auto& old_node : old_level.nodes) {
        int old_node_id = old_node.id;
        
        if (old_node_id == depot_id) continue;
        
        bool found = false;
        
        for (const auto& next_node : next_level.nodes) {
            int new_node_id = next_node.id;
            
            if (new_node_id == depot_id) continue;
            
            auto it = next_level.node_mapping.find(new_node_id);
            if (it != next_level.node_mapping.end()) {
                const vector<int>& next_original_nodes = it->second;
                
                auto old_it = old_level.node_mapping.find(old_node_id);
                if (old_it != old_level.node_mapping.end()) {
                    const vector<int>& old_original_nodes = old_it->second;
                    
                    bool has_overlap = false;
                    for (int old_orig : old_original_nodes) {
                        for (int next_orig : next_original_nodes) {
                            if (old_orig == next_orig) {
                                has_overlap = true;
                                break;
                            }
                        }
                        if (has_overlap) break;
                    }
                    
                    if (has_overlap) {
                        old_to_new_mapping[old_node_id] = new_node_id;
                        found = true;
                        
                        cout << "Old node " << old_node_id << " [";
                        for (int orig : old_original_nodes) cout << orig << " ";
                        cout << "] -> New node " << new_node_id << " [";
                        for (int orig : next_original_nodes) cout << orig << " ";
                        cout << "]" << endl;
                        break;
                    }
                }
            }
        }
        
        if (!found) {
            cerr << "WARNING: Old node " << old_node_id << " not mapped to any next level node!" << endl;
        }
    }
    
    for (size_t v = 0; v < old_sol.route.size(); v++) {    
        int prev_added = -1;
        
        for (int old_node : old_sol.route[v]) {
            auto mapping_it = old_to_new_mapping.find(old_node);
            if (mapping_it == old_to_new_mapping.end()) {
                cerr << "\nERROR: Old node " << old_node << " not found in mapping!" << endl;
                cerr << "Available mappings: ";
                for (const auto& p : old_to_new_mapping) {
                    cerr << p.first << "->" << p.second << " ";
                }
                cerr << endl;
                continue;
            }
            
            int new_node = mapping_it->second;
            cout << new_node << " ";
            
            if (new_node == depot_id) {
                new_sol.route[v].push_back(depot_id);
                prev_added = depot_id;
            } else {
                if (prev_added != new_node) {
                    new_sol.route[v].push_back(new_node);
                    prev_added = new_node;
                } 
            }
        }
        cout << endl;
    }
    
    return new_sol;
}

Solution unmerge_solution_to_previous_level(const Solution& coarse_sol, const LevelInfo& coarse_level, const LevelInfo& fine_level) {
    Solution fine_sol;
    fine_sol.route.resize(coarse_sol.route.size());
    
    map<int, vector<int>> coarse_to_fine_mapping;
    coarse_to_fine_mapping[depot_id] = {depot_id};
    
    for (const auto& coarse_node : coarse_level.nodes) {
        if (coarse_node.id == depot_id) continue;
        
        auto it_coarse = coarse_level.node_mapping.find(coarse_node.id);
        if (it_coarse == coarse_level.node_mapping.end()) continue;
        
        const vector<int>& coarse_original_nodes = it_coarse->second;
        vector<int> corresponding_fine_nodes;
        
        for (int orig : coarse_original_nodes) {
            for (const auto& fine_node : fine_level.nodes) {
                if (fine_node.id == depot_id) continue;
                
                auto it_fine = fine_level.node_mapping.find(fine_node.id);
                if (it_fine != fine_level.node_mapping.end()) {
                    const vector<int>& fine_original_nodes = it_fine->second;
                    
                    if (find(fine_original_nodes.begin(), fine_original_nodes.end(), orig) 
                        != fine_original_nodes.end()) {
                        
                        if (find(corresponding_fine_nodes.begin(), 
                                corresponding_fine_nodes.end(), 
                                fine_node.id) == corresponding_fine_nodes.end()) {
                            corresponding_fine_nodes.push_back(fine_node.id);
                        }
                        break;
                    }
                }
            }
        }
        
        // KIỂM TRA VÀ ÁP DỤNG ORIENTATION
        auto orient_it = merged_node_orientations.find(coarse_node.id);
        if (orient_it != merged_node_orientations.end() && orient_it->second.is_reversed) {
            reverse(corresponding_fine_nodes.begin(), corresponding_fine_nodes.end());
            cout << "Coarse node " << coarse_node.id << " [";
            for (int orig : coarse_original_nodes) cout << orig << " ";
            cout << "] -> Fine nodes [";
            for (int fn : corresponding_fine_nodes) cout << fn << " ";
            cout << "] 🔄 REVERSED" << endl;
        } else {
            cout << "Coarse node " << coarse_node.id << " [";
            for (int orig : coarse_original_nodes) cout << orig << " ";
            cout << "] -> Fine nodes [";
            for (int fn : corresponding_fine_nodes) cout << fn << " ";
            cout << "]" << endl;
        }
        
        coarse_to_fine_mapping[coarse_node.id] = corresponding_fine_nodes;
    }
    
    // Unmerge routes (giữ nguyên phần này)
    for (size_t v = 0; v < coarse_sol.route.size(); v++) {
        cout << "\nVehicle " << v << " coarse route: ";
        for (int id : coarse_sol.route[v]) cout << id << " ";
        cout << endl;
        
        fine_sol.route[v].push_back(depot_id);
        
        for (size_t i = 0; i < coarse_sol.route[v].size(); i++) {
            int coarse_node_id = coarse_sol.route[v][i];
            
            if (coarse_node_id == depot_id) continue;
            
            auto it = coarse_to_fine_mapping.find(coarse_node_id);
            
            if (it != coarse_to_fine_mapping.end()) {
                const vector<int>& fine_nodes = it->second;
                
                cout << "  Coarse node " << coarse_node_id << " -> [";
                for (int fn : fine_nodes) cout << fn << " ";
                cout << "]";
                
                // ✅ HIỂN THỊ ORIENTATION STATUS
                auto orient_it = merged_node_orientations.find(coarse_node_id);
                if (orient_it != merged_node_orientations.end() && orient_it->second.is_reversed) {
                    cout << " 🔄";
                }
                cout << endl;
                
                for (int fine_node_id : fine_nodes) {
                    fine_sol.route[v].push_back(fine_node_id);
                }
            } else {
                cerr << "ERROR: Coarse node " << coarse_node_id 
                     << " not found in coarse_to_fine_mapping!" << endl;
            }
        }
        
        fine_sol.route[v].push_back(depot_id);
        
        cout << "Vehicle " << v << " fine route: ";
        for (int id : fine_sol.route[v]) cout << id << " ";
        cout << endl;
    }
    
    cout << "=== UNMERGING COMPLETE ===" << endl;
    
    return fine_sol;
}

Solution multilevel_tabu_search() {
    Solution s = init_greedy_solution();

    LevelInfo current_level;
    current_level.level_id = 0;
    current_level.nodes.push_back({depot_id, 0.0, 0.0, -1.0, DBL_MAX});
    current_level.nodes.insert(current_level.nodes.end(), C1.begin(), C1.end());
    current_level.nodes.insert(current_level.nodes.end(), C2.begin(), C2.end());

    original_distances = distances;
    current_level.C1_level = C1;
    current_level.C2_level = C2;

    for (const auto& node : current_level.nodes){
        current_level.node_mapping[node.id] = {node.id};
    }

    classify_customers(current_level);
    update_node_index_cache(current_level);
    evaluate_solution(s, &current_level);

    vector<LevelInfo> all_levels;
    all_levels.push_back(current_level);

    int L = 0;
    int max_levels = 4;
    bool coarsening = true;

    double prev_fitness = DBL_MAX;

    while (coarsening && L < max_levels) {
        cout << "\n--- LEVEL " << L << " ---" << endl;   
        update_node_index_cache(all_levels[L]);

        Solution s_current = tabu_search(s, all_levels[L], true);

        if (edge_frequency.empty()){
            update_edge_frequency(s_current);
        }

        for (size_t v = 0; v < s_current.route.size(); v++) {
            cout << "Vehicle " << v << ": ";
            for (int cid : s_current.route[v]) cout << cid << " ";
            cout << endl;
        }
        cout << "Fitness: " << s_current.fitness << endl;
        
        if (L >= 3 && abs(s_current.fitness - prev_fitness) < EPSILON) {
            break;
        }

        prev_fitness = s_current.fitness;
        s = s_current;
        
        LevelInfo next_level = merge_customers(all_levels[L], s, distances, original_distances);
        
        int reduction = all_levels[L].nodes.size() - next_level.nodes.size();
        cout << "Nodes: " << all_levels[L].nodes.size() << " -> " << next_level.nodes.size() << " (reduced " << reduction << ")" << endl;
        
        if (reduction < 1) {
            cout << "Insufficient reduction, stopping coarsening" << endl;
            break;
        }
        
        all_levels.push_back(next_level);
        
        // Project solution
        s = project_solution_to_next_level(s, all_levels[L], next_level);
        for (size_t v = 0; v < s.route.size(); v++) {
            cout << "Vehicle " << v << ": ";
            for (int cid : s.route[v]) {
                cout << cid;
                auto it = next_level.node_mapping.find(cid);
                cout << " ";
            }
            cout << endl;
        }
        
        // Update distances
        vector<vector<double>> next_distances, next_original_distances;
        create_coarse_distance_matrix(next_level, all_levels[L], distances, original_distances,next_distances, next_original_distances);

        cout << "\n=== MATRIX UPDATE ===" << endl;
        cout << "Old: " << distances.size() << "x" << distances[0].size() << endl;
        cout << "New: " << next_distances.size() << "x" << next_distances[0].size() << endl;

        distances = next_distances;
        original_distances = next_original_distances;
        C1 = next_level.C1_level;
        C2 = next_level.C2_level;
        num_nodes = next_level.nodes.size();
        
        update_node_index_cache(next_level);
        // Evaluate projected solution
        evaluate_solution(s, &next_level);
        cout << "Projected solution fitness: " << s.fitness << endl;
        
        L++;
        edge_frequency.clear();
    }
    Solution best_overall = s;
    
    for (int i = 0; i < L; i++) {
        int current_level_id = L - i;
        int prev_level_id = L - i - 1;
        
        cout << "\n=== REFINING FROM LEVEL " << current_level_id << " TO LEVEL " << prev_level_id << " ===" << endl;

        cout << "\n📋 Current Orientation Info:" << endl;
        for (const auto& pair : merged_node_orientations) {
            cout << "  Node " << pair.first 
                 << " (Level " << pair.second.level_id << "): " 
                 << (pair.second.is_reversed ? "REVERSED 🔄" : "NORMAL") << endl;
        }
        
        // Unmerge solution
        s = unmerge_solution_to_previous_level(s, all_levels[current_level_id], all_levels[prev_level_id]);
        
        if (prev_level_id == 0) {
            int n = all_levels[0].nodes.size();
            distances.clear();
            distances.resize(n, vector<double>(n, 0.0));
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        distances[i][j] = sqrt(pow(all_levels[0].nodes[i].x - all_levels[0].nodes[j].x, 2) + pow(all_levels[0].nodes[i].y - all_levels[0].nodes[j].y, 2));
                    }
                }
            }
            original_distances = distances;
        } else {
            cout << "Creating coarse distances for level " << prev_level_id << "..." << endl;
            
            vector<vector<double>> restored_distances, restored_original_distances;

            create_coarse_distance_matrix(
                all_levels[prev_level_id], 
                all_levels[current_level_id],  
                distances,                      
                original_distances,              
                restored_distances,              
                restored_original_distances      
            );
            
            distances = restored_distances;
            original_distances = restored_original_distances;
            
            cout << "Restored distances matrix: " << distances.size() << "x" << distances[0].size() << endl;
        }
        
        C1 = all_levels[prev_level_id].C1_level;
        C2 = all_levels[prev_level_id].C2_level;
        num_nodes = all_levels[prev_level_id].nodes.size();
        
        cout << "Level " << prev_level_id << " stats:" << endl;
        cout << "  Nodes: " << num_nodes << endl;
        cout << "  C1: " << C1.size() << ", C2: " << C2.size() << endl;
        cout << "  Matrix: " << distances.size() << "x" 
             << (distances.empty() ? 0 : distances[0].size()) << endl;
        
        update_node_index_cache(all_levels[prev_level_id]);
        evaluate_solution(s, &all_levels[prev_level_id]);
        cout << "Unmerged fitness: " << s.fitness << endl;
        
        int original_max_iter = MAX_ITER;
        MAX_ITER = max(1000, MAX_ITER / 2); 
        
        edge_frequency.clear();
        s = tabu_search(s, all_levels[prev_level_id], false);
        update_node_index_cache(all_levels[prev_level_id]);
        optimize_all_drone_routes(s, &all_levels[prev_level_id]);
        remove_redundant_depots(s);
        evaluate_solution(s, &all_levels[prev_level_id]);
        
        MAX_ITER = original_max_iter;
        cout << "After tabu: " << s.fitness << endl;
        best_overall = s;
        
        cout << "\nCurrent route at level " << prev_level_id << ":" << endl;
        for (size_t v = 0; v < s.route.size(); v++) {
            cout << "Vehicle " << v << ": ";
            for (int cid : s.route[v]) cout << cid << " ";
            cout << endl;
        }
    }

    return best_overall;
}

Solution create_test_solution_from_routes(const vector<vector<int>>& test_routes) {
    Solution test_sol;
    test_sol.route = test_routes;
    
    cout << "\n" << string(70, '=') << endl;
    cout << "🧪 TESTING WITH PREDEFINED ROUTES" << endl;
    cout << string(70, '=') << "\n" << endl;
    
    // ✅ HIỂN THỊ ROUTES
    for (size_t v = 0; v < test_routes.size(); v++) {
        cout << "Vehicle " << v << " (" 
             << (vehicles[v].is_drone ? "🚁 Drone" : "🚚 Technician") 
             << ", speed=" << vehicles[v].speed << " m/min";
        if (vehicles[v].is_drone) {
            cout << ", limit=" << vehicles[v].limit_drone << " min";
        }
        cout << "): ";
        
        for (int cid : test_routes[v]) {
            cout << cid << " ";
        }
        cout << endl;
    }
    
    // ✅ GỌI HÀM EVALUATE - NÓ ĐÃ TÍNH TẤT CẢ
    evaluate_solution(test_sol, nullptr);
    
    // ✅ HIỂN THỊ KẾT QUẢ
    cout << "\n" << string(70, '=') << endl;
    cout << "📋 TEST RESULTS" << endl;
    cout << string(70, '=') << "\n" << endl;
    
    cout << "Makespan: " << test_sol.makespan << " min" << endl;
    cout << "Drone violation: " << test_sol.drone_violation << " min" << endl;
    cout << "Waiting violation: " << test_sol.waiting_violation << " min" << endl;
    cout << "Fitness: " << test_sol.fitness << endl;
    cout << "Is feasible: " << (test_sol.is_feasible ? "YES ✅" : "NO ❌") << endl;
    
    // ✅ CHI TIẾT VI PHẠM (NẾU CÓ)
    if (!test_sol.is_feasible) {
        cout << "\n⚠️  VIOLATIONS DETECTED:" << endl;
        
        if (test_sol.drone_violation > 0) {
            cout << "  🚁 Drone flight time exceeded by " << test_sol.drone_violation << " min" << endl;
            cout << "     → Some drones flew > " << vehicles[3].limit_drone << " min without returning to depot" << endl;
        }
        
        if (test_sol.waiting_violation > 0) {
            cout << "  ⏳ Customer waiting time exceeded by " << test_sol.waiting_violation << " min" << endl;
            cout << "     → Some customers waited > 60 min for drone to return" << endl;
        }
    } else {
        cout << "\n ALL CONSTRAINTS SATISFIED!" << endl;
    }
    
    return test_sol;
}


int main(int argc, char* argv[]) {
    srand(time(nullptr));

    string dataset_path;
    if (argc > 1) {
        dataset_path = argv[1];
    } else {
        dataset_path = "D:\\New folder\\instances\\50.40.1.txt"; 
    }
    read_dataset(dataset_path);
    printf("MAX_ITER: %d\n", MAX_ITER);
    printf("Segment length: %d\n", SEGMENT_LENGTH);
 
    // Khởi tạo danh sách xe 
    vehicles.clear();
    int customers = num_nodes-1;
    int pairs = 0;
    if (customers >= 6 && customers <= 12) pairs = 1;
    else if (customers <= 20) pairs = 2;
    else if (customers <= 50) pairs = 3;
    else if (customers <= 100) pairs = 4;
    for (int i = 0; i < pairs; ++i) {
        vehicles.push_back({ i+1, 0.58f, false, 0.0f }); // technician
    }
    for (int i = 0; i < pairs; ++i) {
        vehicles.push_back({ pairs + i + 1, 0.83f, true, 120.0f }); // drone
    }

    vector<vector<int>> test_routes = {
        // 3 Technicians
        {0, 38, 4, 8, 26, 1, 11, 30, 9, 12, 0},
        {0, 14, 31, 47, 13, 32, 24, 46, 0},
        {0, 29, 22, 7, 5, 0},
        
        // 3 Drones
        {0, 16, 2, 6, 28, 35, 20, 27, 39, 36, 50, 23, 0, 17, 0},
        {0, 45, 18, 3, 19, 48, 15, 44, 34, 33, 10, 0},
        {0, 21, 25, 40, 42, 41, 37, 43, 49, 0}
    };

    //Solution test_solution = create_test_solution_from_routes(test_routes);

    Solution best_solution = multilevel_tabu_search();
    print_solution(best_solution);

    return 0;
}



