#include<bits/stdc++.h>
using namespace std;

struct Node {
    int id;
    double x,y;
    double c1_or_c2;
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
    double fitness; // giá trị hàm mục tiêu
    bool is_feasible; // lời giải có hợp lệ không

    Solution(): makespan(0), drone_violation(0), fitness(DBL_MAX), is_feasible(true) {}
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

struct LevelInfo {
    vector<Node> nodes;
    vector<Node> C1_level, C2_level; // customers ở level này
    map<int, vector<int>> node_mapping; // ánh xạ từ node level này về node gốc
    vector<vector<double>> truck_time_matrix; // ma trận thời gian cho technician
    vector<vector<double>> drone_time_matrix; // ma trận thời gian cho drone
    int level_id;
    int num_customers;

    LevelInfo() : level_id(0), num_customers(0) {}
};

struct MergedNodeInfo {
    int merged_node_id;
    vector<int> original_sequence;  // Thứ tự nodes trong group: [17, 13, 15]
    vector<int> current_sequence;   // Sequence ở level hiện tại
    double internal_truck_time;     // Tổng thời gian truck bên trong
    double internal_drone_time;     // Tổng thời gian drone bên trong
    vector<double> cumulative_truck_times;
    vector<double> cumulative_drone_times;
    int entry_node_original;                 // Node đầu tiên (entry point)
    int exit_node_original;                  // Node cuối cùng (exit point)
    int entry_node;
    int exit_node;
    int level_id;
    
    MergedNodeInfo() : merged_node_id(-1), internal_truck_time(0.0), internal_drone_time(0.0), entry_node_original(-1), exit_node_original(-1), entry_node(-1), exit_node(-1), level_id(-1) {}
};

vector<vector<double>> base_distance_matrix;
vector<vector<double>> truck_times;
vector<vector<double>> drone_times;
vector<Node> C1; // customers served only by technicians
vector<Node> C2; // customers served by drones or technicians
vector<VehicleFamily> vehicles;
map<int, MergedNodeInfo> merged_nodes_info;
unordered_map<int, int> base_type_by_node;

constexpr double TRUCK_SPEED = 0.58;
constexpr double DRONE_SPEED = 0.83;

int depot_id = 0;
int num_nodes = 0;
double alpha1 = 1.0; // tham số hàm phạt thứ nhất
double Beta = 0.5; // tham số điều chỉnh hệ số hàm phạt

int MAX_ITER;
int TABU_TENURE;
int MAX_NO_IMPROVE;
double EPSILON = 1e-6;

int MAX_LEVELS = 4;
bool USE_MANUAL_SEGMENT_CONFIG = false;
double MERGE_RATIO = 0.10;

// Adaptive parameters
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

void build_time_matrices_from_distance(const vector<vector<double>>& distance_matrix,
                                       vector<vector<double>>& truck_time_matrix,
                                       vector<vector<double>>& drone_time_matrix) {
    const int n = static_cast<int>(distance_matrix.size());
    truck_time_matrix.assign(n, vector<double>(n, 0.0));
    drone_time_matrix.assign(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            truck_time_matrix[i][j] = distance_matrix[i][j] / TRUCK_SPEED;
            drone_time_matrix[i][j] = distance_matrix[i][j] / DRONE_SPEED;
        }
    }
}

void read_dataset(const string &filename){
    vector<Node> nodes;
    C1.clear();
    C2.clear();
    base_type_by_node.clear();
    ifstream file(filename);
    if (!file.is_open()){
        cerr << "Error opening file: " << filename <<endl;
        exit(1);
    }
    nodes.push_back({depot_id,0.0,0.0,-1.0}); // depot
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
    if (nodes.size() > 1000) {
        // Bộ rất lớn (> 1000)
        MAX_ITER = 50000;
        MAX_NO_IMPROVE = 500000;
    }
    else if (nodes.size() >= 1000) {
        // Bộ 1000 (501-1000)
        MAX_ITER = 25000;
        MAX_NO_IMPROVE = 500000;
    }
    else if (nodes.size() >= 500) {
        // Bộ 500 (201-500)
        MAX_ITER = 12500;
        MAX_NO_IMPROVE = 500000;
    }
    else if (nodes.size() >= 200) {
        // Bộ 200 (101-200)
        MAX_ITER = 6000;
        MAX_NO_IMPROVE = 500000;
    }
    else if (nodes.size() >= 100) {
        // Bộ 100 (100)
        MAX_ITER = 1000;
        MAX_NO_IMPROVE = 500000;
    }
    else if (nodes.size() >= 50) {
        // Bộ 50 (50-99)
        MAX_ITER = 500;
        MAX_NO_IMPROVE = 500000;
    }
    else {
        // Bộ nhỏ (6-49)
        MAX_ITER = 500;
        MAX_NO_IMPROVE = 500000;
    }

    for (const auto& node : nodes) {
        if (node.id == depot_id) {
            cout << "Node id: " << node.id << " (depot), x: " << node.x << ", y: " << node.y << endl;
            continue;
        } else {
            cout << "Node id: " << node.id << ", x: " << node.x << ", y: " << node.y
                 << ", type: " << (node.c1_or_c2 > 0 ? "C2" : "C1") << endl;
        }
    }

    // Tính ma trận khoảng cách hình học để suy ra ma trận thời gian.
    base_distance_matrix.resize(nodes.size(), vector<double>(nodes.size(), 0));
    for (size_t i = 0; i < nodes.size(); ++i){
        for (size_t j = 0; j < nodes.size(); ++j){
            if (i != j){
                base_distance_matrix[i][j] = sqrt(pow(nodes[i].x - nodes[j].x, 2) + pow(nodes[i].y - nodes[j].y, 2));
            }
        }
    }
    build_time_matrices_from_distance(base_distance_matrix, truck_times, drone_times);

    // Phân loại khách hàng
    for (const auto& node : nodes){
        if (node.id == depot_id) continue;
        if (node.c1_or_c2 > 0){
            C2.push_back(node);
            base_type_by_node[node.id] = 2;
        } else if (node.c1_or_c2 == 0) {
            C1.push_back(node);
            base_type_by_node[node.id] = 1;
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
    cout << "Fitness: " << sol.fitness << endl;
}

unordered_map<int, int> node_id_to_index_cache;

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

void normalize_route(vector<int> &route) {
    if (route.empty()) { route.push_back(depot_id); return; }
    vector<int> tmp;
    tmp.reserve(route.size());
    // đảm bảo bắt đầu bằng depot
    if (route.front() != depot_id) tmp.push_back(depot_id);
    for (int x : route) {
        // bỏ depot liên tiếp
        if (!tmp.empty() && tmp.back() == depot_id && x == depot_id) continue;
        tmp.push_back(x);
    }
    // đảm bảo chỉ một depot ở cuối
    if (tmp.empty() || tmp.back() != depot_id) tmp.push_back(depot_id);
    route.swap(tmp);
}

void evaluate_solution(Solution &sol, const LevelInfo *current_level = nullptr) {
    for (auto &route : sol.route) normalize_route(route);

    sol.makespan = 0;
    sol.drone_violation = 0;
    sol.fitness = 0;
    sol.is_feasible = true;

    for (size_t i = 0; i < sol.route.size(); i++){
        const bool is_drone = vehicles[i].is_drone;
        const vector<vector<double>>& active_time_matrix =
            (current_level != nullptr)
                ? (is_drone ? current_level->drone_time_matrix : current_level->truck_time_matrix)
                : (is_drone ? drone_times : truck_times);

        int prev = depot_id;
        double current_time = 0;
        double depart_time = 0;

        for (int j = 0; j < sol.route[i].size(); j++) {
            int cid = sol.route[i][j];
            
            if (cid == depot_id){
                if (prev != depot_id){
                    double travel_time = 0.0;
                    
                    if (current_level != nullptr) {
                        int prev_idx = find_node_index_fast(prev);
                        int depot_idx = find_node_index_fast(depot_id);
                        if (prev_idx != -1 && depot_idx != -1) {
                            travel_time = active_time_matrix[prev_idx][depot_idx];
                        }
                    } else {
                        travel_time = active_time_matrix[prev][depot_id];
                    }
                    current_time += travel_time;
                }
                
                double arrival_depot = current_time;
                double flight_time = arrival_depot - depart_time;
                
                if (vehicles[i].is_drone){
                    sol.drone_violation += max(0.0, flight_time - vehicles[i].limit_drone);
                }
                
                if (sol.drone_violation > 0 ) {
                    sol.is_feasible = false;
                }
                
                depart_time = current_time;
                prev = depot_id;
            } else {
                double travel_time = 0.0;
                double internal_time = 0.0;
                
                if (current_level != nullptr) {
                    int prev_idx = find_node_index_fast(prev);
                    int cid_idx = find_node_index_fast(cid);

                    if (prev_idx != -1 && cid_idx != -1) {
                        travel_time = active_time_matrix[prev_idx][cid_idx];
                        auto info_it = merged_nodes_info.find(cid);
                        if (info_it != merged_nodes_info.end()) {
                            internal_time = is_drone ? info_it->second.internal_drone_time : info_it->second.internal_truck_time;
                            travel_time += internal_time;
                        }
                    } 
                } else {
                    travel_time = active_time_matrix[prev][cid];
                }

                current_time += travel_time;
                prev = cid;
            }
        }
        sol.makespan = max(sol.makespan, current_time);
    }

    sol.fitness = sol.makespan + alpha1*sol.drone_violation ;
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
        // Dùng cache từ dataset gốc
        auto it = base_type_by_node.find(nid);
        if (it != base_type_by_node.end()) return it->second;
    }
    return -1;
}

Solution init_greedy_solution() {
    auto start_time = chrono::high_resolution_clock::now();
    Solution sol;
    sol.route.resize(vehicles.size());

    for (size_t v = 0; v < vehicles.size(); ++v)
        sol.route[v].push_back(depot_id);

    // 1. GÁN C1 CHO TECHNICIAN 
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
        double best_time = DBL_MAX;
        int best_tech = -1, best_cid = -1, best_idx = -1;
        for (size_t t = 0; t < tech_indices.size(); t++) {
            for (size_t i = 0; i < unserved_C1.size(); i++) {
                int cid = unserved_C1[i];
                double d = truck_times[tech_pos[t]][cid];
                if (d < best_time) {
                    best_time = d;
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

    vector<int> unserved_C2;
    for (const auto& n : C2) unserved_C2.push_back(n.id);

    int total_vehicles = vehicles.size();
    int customers_per_vehicle = unserved_C2.size() / total_vehicles;
    int extra_customers = unserved_C2.size() % total_vehicles;

    vector<int> vehicle_quota(total_vehicles);
    for (int v = 0; v < total_vehicles; v++) {
        vehicle_quota[v] = customers_per_vehicle;
        if (v < extra_customers) vehicle_quota[v]++; 
    }

    cout << "\n📊 CUSTOMER ALLOCATION:" << endl;
    for (size_t v = 0; v < vehicles.size(); v++) {
        cout << "Vehicle " << v << " (" 
             << (vehicles[v].is_drone ? "Drone" : "Tech") 
             << "): " << vehicle_quota[v] << " customers" << endl;
    }

    vector<int> current_pos(vehicles.size());
    vector<int> assigned_count(vehicles.size(), 0);

    for (size_t v = 0; v < vehicles.size(); v++) {
        if (!sol.route[v].empty() && sol.route[v].back() != depot_id) {
            current_pos[v] = sol.route[v].back();
        } else {
            current_pos[v] = depot_id;
        }
    }

    while (!unserved_C2.empty()) {
        double best_time = DBL_MAX;
        int best_vehicle = -1;
        int best_cid_idx = -1;

        // Tìm customer gần nhất cho từng xe 
        for (size_t v = 0; v < vehicles.size(); v++) {
            // Bỏ qua nếu đã đủ quota
            if (assigned_count[v] >= vehicle_quota[v]) continue;

            for (size_t i = 0; i < unserved_C2.size(); i++) {
                int cid = unserved_C2[i];
                
                // Drone không được gán C1
                if (vehicles[v].is_drone && get_type(cid, nullptr) == 1) {
                    continue;
                }

                const vector<vector<double>>& time_matrix = vehicles[v].is_drone ? drone_times : truck_times;
                double dist = time_matrix[current_pos[v]][cid];

                if (dist < best_time) {
                    best_time = dist;
                    best_vehicle = v;
                    best_cid_idx = i;
                }
            }
        }

        // Gán customer cho xe
        if (best_vehicle != -1) {
            int best_cid = unserved_C2[best_cid_idx];

            sol.route[best_vehicle].push_back(best_cid);
            current_pos[best_vehicle] = best_cid;
            assigned_count[best_vehicle]++;

            unserved_C2.erase(unserved_C2.begin() + best_cid_idx);
        } else {
            cout << "⚠️  WARNING: Cannot assign " << unserved_C2.size() 
                 << " remaining customers!" << endl;
            break;
        }
    }

    for (size_t v = 0; v < vehicles.size(); v++) {
        if (sol.route[v].back() != depot_id) {
            sol.route[v].push_back(depot_id);
        }
    }


    evaluate_solution(sol);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "\n⏱️  Initial solution generated in " << elapsed.count() << " seconds." << endl;
    cout << "\n📊 INITIAL SOLUTION:" << endl;
    print_solution(sol);

    return sol;
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

    if (pos1 == 0 || pos1 == new_sol.route[v1].size() - 1) {
        return current_sol;
    }

    if (pos2 == new_sol.route[v2].size() && vehicles[v2].is_drone){
        if (get_type(cid, current_level) == 1) return current_sol; 
        int customer_count = 0;
        for (int node : new_sol.route[v1]) {
            if (node != depot_id) customer_count++;
        }
        if (customer_count <= 1 && v1 == v2) {
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].push_back(cid);
        if (!new_sol.route[v2].empty() && new_sol.route[v2].back() != depot_id) {
            new_sol.route[v2].push_back(depot_id);
        }
    } else {
        if (v1 == v2) return current_sol;
        if (pos2 == 0 || pos2 >= new_sol.route[v2].size()) return current_sol;
        if (get_type(cid, current_level) == 1 && vehicles[v2].is_drone) {
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid);
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

    if (pos2 == new_sol.route[v2].size() && vehicles[v2].is_drone){
        if (get_type(cid1, current_level) == 1 || get_type(cid2, current_level) == 1){
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].push_back(cid1);
        new_sol.route[v2].push_back(cid2);
        new_sol.route[v2].push_back(depot_id);
    } else {
        if ((get_type(cid1, current_level) == 1 || get_type(cid2, current_level) == 1) && vehicles[v2].is_drone){
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid1);
        new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2 + 1, cid2);
    }

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
        
        reverse(new_sol.route[v1].begin() + pos1, new_sol.route[v1].begin() + pos2 + 1);
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

Solution tabu_search(Solution initial_sol, const LevelInfo *current_level, bool track_edge = true){
    if (current_level != nullptr) update_node_index_cache(*current_level);

    Solution best_sol = initial_sol;
    Solution current_sol = initial_sol;

    vector<TabuMove> tabu_list; // danh sách các move bị tabu
    int no_improve_count = 0;
    int last_depot_opt_iter = 0;
    int no_improve_segment_length = 0;
    const int max_no_improve_segment = 8;

    vector<string> move_types = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
    
    for (int iter = 0; iter < MAX_ITER && no_improve_count < MAX_NO_IMPROVE; iter++){
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
                    if (customer_count_v1 <= 1) continue; 

                    for (size_t v2 = 0; v2 < current_sol.route.size(); v2++) {
                        if (v1 == v2) continue;
                        for (size_t pos2 = 1; pos2 <= current_sol.route[v2].size(); pos2++) {
                            if (pos2 == current_sol.route[v2].size()){
                                if (!vehicles[v2].is_drone) continue;
                                if (get_type(n1, current_level) == 1) continue;
                                if (v1 == v2) continue;
                            } else {
                                if (v1 == v2) continue;
                                if (pos2 == current_sol.route[v2].size() - 1) continue;
                                if (get_type(n1, current_level) == 1 && vehicles[v2].is_drone) continue;
                            }

                            Solution new_sol = move_1_0(current_sol, v1, pos1, v2, pos2, current_level);
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
                            if (n2 == depot_id || n1 == n2 || get_type(n1, current_level) != get_type(n2, current_level) || ((abs(int(pos1)-int(pos2)) <= 1) && (v1 == v2))) continue;

                            Solution new_sol = move_1_1(current_sol, v1, pos1, v2, pos2, current_level);
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
                        if ((get_type(n1, current_level) == 1 || get_type(n2, current_level) == 1) && vehicles[v2].is_drone) continue;
                        for (size_t pos2 = 1; pos2 <= current_sol.route[v2].size(); pos2++){
                            if (pos2 == current_sol.route[v2].size() && !vehicles[v2].is_drone) {
                                continue;
                            }

                            Solution new_sol = move_2_0(current_sol, v1, pos1, v2, pos2, current_level);
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
                            if ((get_type(n1, current_level) == 1 || get_type(n2, current_level) == 1) && vehicles[v2].is_drone) continue;
                            if (get_type(n3, current_level) == 1 && vehicles[v1].is_drone) continue;
                            Solution new_sol = move_2_1(current_sol, v1, pos1, v2, pos2, current_level);
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
                            if ((get_type(n1, current_level) == 1 || get_type(n2, current_level) == 1) && vehicles[v2].is_drone) continue;
                            if ((get_type(n3, current_level) == 1 || get_type(n4, current_level) == 1) && vehicles[v1].is_drone) continue;

                            Solution new_sol = move_2_2(current_sol, v1, pos1, v2, pos2, current_level);
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

                        Solution new_sol = move_2opt(current_sol, v1, pos1, v1, pos2, current_level); // Cùng xe v1
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
                                    if (cid != depot_id && get_type(cid, current_level) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (!invalid_move && vehicles[v1].is_drone) {
                                for (size_t i = pos2; i < current_sol.route[v2].size() - 1; i++) {
                                    int cid = current_sol.route[v2][i];
                                    if (cid != depot_id && get_type(cid, current_level) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (invalid_move) continue;

                            int customer_at_pos1 = current_sol.route[v1][pos1];
                            int customer_at_pos2 = current_sol.route[v2][pos2];
                            
                            Solution new_sol = move_2opt(current_sol, v1, pos1, v2, pos2, current_level); // Khác xe v1 và v2
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
            evaluate_solution(current_sol, current_level);

            /*cout << "Iter: " << iter << " Move: " << move_type 
                 << " current makespan: " << current_sol.makespan 
                 << ", drone_violation: " << current_sol.drone_violation 
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

        update_weights();
    }
    return best_sol;
}

void create_coarse_distance_matrix(LevelInfo& next_level, const LevelInfo& current_level,const vector<vector<double>>& curr_distances,vector<vector<double>>& next_distances){   
    update_node_index_cache(current_level);
    int n = next_level.nodes.size();
    next_distances.resize(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (i == j){
                next_distances[i][j] = 0.0;
                continue;
            }

            int node_i_id = next_level.nodes[i].id;
            int node_j_id = next_level.nodes[j].id;

            // check xem có phải merge node không
            auto it_i = merged_nodes_info.find(node_i_id);
            auto it_j = merged_nodes_info.find(node_j_id);

            bool i_is_merged = (it_i != merged_nodes_info.end());
            bool j_is_merged = (it_j != merged_nodes_info.end());

            int departure_node, arrival_node;
            if (i_is_merged){
                departure_node = it_i->second.exit_node;
            } else {
                departure_node = node_i_id;
            }

            if (j_is_merged){
                arrival_node = it_j->second.entry_node;
            } else {
                arrival_node = node_j_id;
            }
            int dep_idx = find_node_index_fast(departure_node);
            int arr_idx = find_node_index_fast(arrival_node);
            double distance = 0.0;
            if (dep_idx != -1 && arr_idx != -1){
                distance = curr_distances[dep_idx][arr_idx];
            } else {
                cout << "WARNING: dep_idx=" << dep_idx << " arr_idx=" << arr_idx
                 << " for departure_node=" << departure_node << " arrival_node=" << arrival_node
                 << " (i=" << i << ", j=" << j << ")" << endl;
            }
            next_distances[i][j] = distance;
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
        
        int idx_from = find_node_index_fast(from_node);
        int idx_to = find_node_index_fast(to_node);
        if (idx_from == -1 || idx_to == -1) {
            continue;
        }

        const Node& node_from = current_level.nodes[idx_from];
        const Node& node_to = current_level.nodes[idx_to];
        
        bool same_type = (node_from.c1_or_c2 == 0 && node_to.c1_or_c2 == 0) || 
                        (node_from.c1_or_c2 > 0 && node_to.c1_or_c2 > 0);
        
        if (same_type){
            candidates.emplace_back(make_tuple(frequency, from_node, to_node));
            /*cout << "Candidate edge: (" << from_node << ", " << to_node 
                 << ") frequency=" << frequency  << endl;*/
        }
    }

    sort(candidates.begin(), candidates.end(), 
         [](const tuple<int, int, int>& a, const tuple<int, int, int>& b) {
        return get<0>(a) > get<0>(b);
    });
    
    return candidates;
}

LevelInfo merge_customers(const LevelInfo& current_level,
                         const Solution& best_solution,
                         const vector<vector<double>>& curr_truck_times,
                         const vector<vector<double>>& curr_drone_times) {
    LevelInfo next_level;
    next_level.level_id = current_level.level_id + 1;
    update_node_index_cache(current_level);
    
    vector<tuple<int,int,int>> candidates = collect_merge_candidates(current_level, best_solution);
    
    // Số cạnh được chọn để merge theo tỉ lệ nén cấu hình
    int num_to_merge = max(1, (int)(candidates.size() * MERGE_RATIO));
    
    //cout << "\n=== MERGING " << num_to_merge << " / " << candidates.size() << " EDGES (20%) ===" << endl;
    
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
            //cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b << ") freq=" << frequency << " → NEW GROUP" << endl;
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
                cout << " Warning: node " << node_a << " not at group ends!" << endl;
                continue;
            }
            merged_nodes.insert(node_b);
            /*cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b 
                 << ") freq=" << frequency << " → ADD TO GROUP " << group_idx_a << endl;*/
        }
        // Case 3: node_b đã có group, node_a chưa -> Thêm node_a vào group của node_b
        else if (group_idx_a == -1 && group_idx_b != -1) {
            if (merged_groups[group_idx_b].front() == node_b){
                merged_groups[group_idx_b].insert(merged_groups[group_idx_b].begin(), node_a);
            } else if (merged_groups[group_idx_b].back() == node_b){
                merged_groups[group_idx_b].push_back(node_a);
            } else {
                // Không nên xảy ra
                continue;
            }
            merged_nodes.insert(node_a);
            /*cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b 
                 << ") freq=" << frequency << " → ADD TO GROUP " << group_idx_b << endl;*/
        }
        // Case 4: Cả 2 đã có group khác nhau → Merge 2 groups
        else if (group_idx_a != group_idx_b) {
            // Chỉ nối nếu node_a ở cuối group_a VÀ node_b ở đầu group_b
            if (merged_groups[group_idx_a].back() == node_a && merged_groups[group_idx_b].front() == node_b) {
                // Nối group_b vào cuối group_a
                merged_groups[group_idx_a].insert(
                    merged_groups[group_idx_a].end(),
                    merged_groups[group_idx_b].begin(),
                    merged_groups[group_idx_b].end()
                );
                merged_groups.erase(merged_groups.begin() + group_idx_b);
                /*cout << "Edge " << (i+1) << ": (" << node_a << " → " << node_b 
                     << ") freq=" << frequency << " → CONNECT GROUPS" << endl;*/
            } else {
                cout << " Cannot connect - nodes not at boundaries" << endl;
            }
        }
    }
    
    //cout << "\n=== FINAL MERGED GROUPS ===" << endl;
    for (size_t i = 0; i < merged_groups.size(); i++) {
        cout << "Group " << (i+1) << ": [";
        for (size_t j = 0; j < merged_groups[i].size(); j++) {
            cout << merged_groups[i][j];
            if (j < merged_groups[i].size() - 1) cout << " -> ";
        }
        cout << "]" << endl;
    }
    
    if (merged_groups.empty()) {
        cout << " No groups formed! Returning current level." << endl;
        return current_level;
    }
    
    // Đặt tên mới cho node
    int next_node_id = (next_level.level_id) * 1000;
    
    next_level.nodes.push_back({depot_id, 0.0, 0.0, -1.0});
    next_level.node_mapping[depot_id] = {depot_id};
    
    for (const auto& group : merged_groups) {
        int first_node_id = group[0];
        int idx = find_node_index_fast(first_node_id);
        
        if (idx != -1) {
            const Node& first_node = current_level.nodes[idx];
            Node merged_node = {
                next_node_id++,
                0.0,
                0.0,
                first_node.c1_or_c2,
            };
            next_level.nodes.push_back(merged_node);

            MergedNodeInfo info;
            info.merged_node_id = merged_node.id;
            info.level_id = next_level.level_id;
            info.current_sequence = group;
            info.entry_node = group.front();
            info.exit_node = group.back();
            info.internal_truck_time = 0.0;
            info.internal_drone_time = 0.0;
            info.cumulative_truck_times.resize(group.size(), 0.0);
            info.cumulative_drone_times.resize(group.size(), 0.0);
            double cumulative_truck = 0.0;
            double cumulative_drone = 0.0;
            
            for (size_t i = 0; i < group.size(); i++) {
                info.cumulative_truck_times[i] = cumulative_truck;
                info.cumulative_drone_times[i] = cumulative_drone;
                if (i < group.size() - 1) {
                    int from = group[i];
                    int to = group[i + 1];
                    int idx_from = find_node_index_fast(from);
                    int idx_to = find_node_index_fast(to);
                    if (idx_from != -1 && idx_to != -1) {
                        double truck_edge_time = curr_truck_times[idx_from][idx_to];
                        double drone_edge_time = curr_drone_times[idx_from][idx_to];
                        cumulative_truck += truck_edge_time;
                        cumulative_drone += drone_edge_time;
                        info.internal_truck_time += truck_edge_time;
                        info.internal_drone_time += drone_edge_time;
                    }
                    auto it_merge = merged_nodes_info.find(from);
                    if (it_merge != merged_nodes_info.end()) {
                        cumulative_truck += it_merge->second.internal_truck_time;
                        cumulative_drone += it_merge->second.internal_drone_time;
                        info.internal_truck_time += it_merge->second.internal_truck_time;
                        info.internal_drone_time += it_merge->second.internal_drone_time;
                    }
                }
            }
            int exit_node = group.back();
            auto it_exit = merged_nodes_info.find(exit_node);
            if (it_exit != merged_nodes_info.end()) {
                info.internal_truck_time += it_exit->second.internal_truck_time;
                info.internal_drone_time += it_exit->second.internal_drone_time;
            }
            
            // ánh xạ node merge về toàn bộ node gốc
            vector<int> original_nodes;
            for (int node_id : group) {
                auto it = current_level.node_mapping.find(node_id);
                if (it != current_level.node_mapping.end()) {
                    for (int orig : it->second) {
                        if (find(original_nodes.begin(), original_nodes.end(), orig) == original_nodes.end()) {
                            original_nodes.push_back(orig);
                        }
                    }
                } else {
                    if (find(original_nodes.begin(), original_nodes.end(), node_id) == original_nodes.end()) {
                        original_nodes.push_back(node_id);
                    }
                }
            }
            info.original_sequence = original_nodes;
            next_level.node_mapping[merged_node.id] = original_nodes;
            merged_nodes_info[merged_node.id] = info;
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

    int n = next_level.nodes.size();
    next_level.truck_time_matrix.resize(n, vector<double>(n, 0.0));
    next_level.drone_time_matrix.resize(n, vector<double>(n, 0.0));

    auto lookup_transition_time = [&](int node_id_i, int node_id_j, const vector<vector<double>>& curr_matrix) {
        int idx_i = find_node_index_fast(node_id_i);
        int idx_j = find_node_index_fast(node_id_j);
        if (idx_i != -1 && idx_j != -1) {
            return curr_matrix[idx_i][idx_j];
        }
        if (idx_i != -1 && idx_j == -1) {
            auto it_j = merged_nodes_info.find(node_id_j);
            if (it_j != merged_nodes_info.end()) {
                int idx_entry_j = find_node_index_fast(it_j->second.entry_node);
                if (idx_entry_j != -1) return curr_matrix[idx_i][idx_entry_j];
            }
            return 0.0;
        }
        if (idx_i == -1 && idx_j != -1) {
            auto it_i = merged_nodes_info.find(node_id_i);
            if (it_i != merged_nodes_info.end()) {
                int idx_exit_i = find_node_index_fast(it_i->second.exit_node);
                if (idx_exit_i != -1) return curr_matrix[idx_exit_i][idx_j];
            }
            return 0.0;
        }

        auto it_i = merged_nodes_info.find(node_id_i);
        auto it_j = merged_nodes_info.find(node_id_j);
        if (it_i != merged_nodes_info.end() && it_j != merged_nodes_info.end()) {
            int idx_exit_i = find_node_index_fast(it_i->second.exit_node);
            int idx_entry_j = find_node_index_fast(it_j->second.entry_node);
            if (idx_exit_i != -1 && idx_entry_j != -1) {
                return curr_matrix[idx_exit_i][idx_entry_j];
            }
        }
        return 0.0;
    };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                next_level.truck_time_matrix[i][j] = 0.0;
                next_level.drone_time_matrix[i][j] = 0.0;
                continue;
            }
            int node_id_i = next_level.nodes[i].id;
            int node_id_j = next_level.nodes[j].id;
            next_level.truck_time_matrix[i][j] = lookup_transition_time(node_id_i, node_id_j, curr_truck_times);
            next_level.drone_time_matrix[i][j] = lookup_transition_time(node_id_i, node_id_j, curr_drone_times);
        }
    }
    
    /*cout << "\n Level " << next_level.level_id << " created: " 
         << current_level.nodes.size() << " → " << next_level.nodes.size() 
         << " nodes (merged " << (current_level.nodes.size() - next_level.nodes.size()) 
         << ")" << endl;*/
    
    return next_level;
}

Solution project_solution_to_next_level(const Solution& old_sol, const LevelInfo& old_level, const LevelInfo& next_level){
    Solution new_sol;
    new_sol.route.resize(old_sol.route.size());
    update_node_index_cache(next_level);
    
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
    }
    
    return new_sol;
}

Solution unmerge_solution_to_previous_level(const Solution& coarse_sol, const LevelInfo& coarse_level, const LevelInfo& fine_level) {
    Solution fine_sol;
    fine_sol.route.resize(coarse_sol.route.size());

    set<int> fine_level_node_ids;
    for (const auto& node : fine_level.nodes) {
        fine_level_node_ids.insert(node.id);
    }
    
    map<int, vector<int>> coarse_to_fine;
    coarse_to_fine[depot_id] = {depot_id};
    
    for (const auto& coarse_node : coarse_level.nodes) {
        if (coarse_node.id == depot_id) continue;
        
        auto info_it = merged_nodes_info.find(coarse_node.id);
        
        if (info_it != merged_nodes_info.end() && 
            info_it->second.level_id == coarse_level.level_id) {
            
            const MergedNodeInfo& info = info_it->second;
            
            /*cout << "  UNMERGE " << coarse_node.id << " → [";
            for (size_t i = 0; i < info.current_sequence.size(); i++) {
                cout << info.current_sequence[i];
                if (i < info.current_sequence.size() - 1) cout << ", ";
            }
            cout << "]" << endl;*/
            
            coarse_to_fine[coarse_node.id] = info.current_sequence;
        } else if (fine_level_node_ids.find(coarse_node.id) != fine_level_node_ids.end()) {
            //cout << "  KEEP " << coarse_node.id << " (exists in fine level)" << endl;
            coarse_to_fine[coarse_node.id] = {coarse_node.id};
        } else {
            /*cout << "  WARNING: Node " << coarse_node.id 
                 << " not found in fine level and not merged at current level!" << endl;*/
            if (info_it != merged_nodes_info.end()) {
                /*cout << "  FORCE UNMERGE " << coarse_node.id 
                     << " (merged at level " << info_it->second.level_id << ") → [";
                for (size_t i = 0; i < info_it->second.current_sequence.size(); i++) {
                    cout << info_it->second.current_sequence[i];
                    if (i < info_it->second.current_sequence.size() - 1) cout << ", ";
                }
                cout << "]" << endl;*/
                
                coarse_to_fine[coarse_node.id] = info_it->second.current_sequence;
            } else {
                coarse_to_fine[coarse_node.id] = {coarse_node.id};
            }
        }
    }

    // UNMERGE ROUTES: giu nguyen thu tu route coarse, thay moi node bang chuoi fine tuong ung.
    for (size_t v = 0; v < coarse_sol.route.size(); v++) {
        for (int coarse_node_id : coarse_sol.route[v]) {
            auto it = coarse_to_fine.find(coarse_node_id);
            if (it == coarse_to_fine.end()) {
                continue;
            }
            for (int fine_node_id : it->second) {
                fine_sol.route[v].push_back(fine_node_id);
            }
        }
        normalize_route(fine_sol.route[v]);
    }
    
    return fine_sol;
}

Solution multilevel_tabu_search() {
    auto total_start = chrono::high_resolution_clock::now();
    Solution s = init_greedy_solution();

    LevelInfo current_level;
    current_level.level_id = 0;
    current_level.nodes.push_back({depot_id, 0.0, 0.0, -1.0});
    current_level.nodes.insert(current_level.nodes.end(), C1.begin(), C1.end());
    current_level.nodes.insert(current_level.nodes.end(), C2.begin(), C2.end());

    current_level.C1_level = C1;
    current_level.C2_level = C2;

    for (const auto& node : current_level.nodes){
        current_level.node_mapping[node.id] = {node.id};
    }

    current_level.truck_time_matrix = truck_times;
    current_level.drone_time_matrix = drone_times;

    classify_customers(current_level);
    update_node_index_cache(current_level);
    evaluate_solution(s, &current_level);

    vector<LevelInfo> all_levels;
    all_levels.push_back(current_level);

    int L = 0;
    int max_levels = MAX_LEVELS;
    bool coarsening = true;
    double prev_fitness = DBL_MAX;

    while (coarsening && L < max_levels) {
        //cout << "\n--- LEVEL " << L << " ---" << endl;
        auto level_start = chrono::high_resolution_clock::now();   
        update_node_index_cache(all_levels[L]);
        Solution s_current = tabu_search(s, &all_levels[L], true);
        update_edge_frequency(s_current);
        auto level_end = chrono::high_resolution_clock::now();
        double level_time = chrono::duration<double>(level_end - level_start).count();
        print_solution(s_current);
        if (L >= 3 && s_current.fitness == prev_fitness) {
            break;
        }
        prev_fitness = s_current.fitness;
        s = s_current;
        auto merge_start = chrono::high_resolution_clock::now();
        LevelInfo next_level = merge_customers(all_levels[L], s, truck_times, drone_times);
        //cout << "Nodes in next_level: ";
        //for (const auto& node : next_level.nodes) cout << node.id << " ";
        //cout << endl;
        auto merge_end = chrono::high_resolution_clock::now();
        double merge_time = chrono::duration<double>(merge_end - merge_start).count();
        int reduction = all_levels[L].nodes.size() - next_level.nodes.size();
        cout << "⏱️  Level " << all_levels[L].level_id << " Merging Time: " 
             << fixed << setprecision(10) << merge_time << "s" << endl;
        if (reduction < 1) {
            cout << "Insufficient reduction, stopping coarsening" << endl;
            break;
        }
        all_levels.push_back(next_level);

        truck_times = next_level.truck_time_matrix;
        drone_times = next_level.drone_time_matrix;
        C1 = next_level.C1_level;
        C2 = next_level.C2_level;
        num_nodes = next_level.nodes.size();
        
        // Project solution
        auto project_start = chrono::high_resolution_clock::now();
        s = project_solution_to_next_level(s, all_levels[L], next_level);
        auto project_end = chrono::high_resolution_clock::now();
        double project_time = chrono::duration<double>(project_end - project_start).count();
        cout << "⏱️  Level " << all_levels[L].level_id << " Projection Time: " 
             << fixed << setprecision(10) << project_time << "s" << endl;
        update_node_index_cache(next_level);
        for (size_t v = 0; v < s.route.size(); v++) {
            cout << "Vehicle " << v << ": ";
            for (int cid : s.route[v]) {
                cout << cid;
                cout << " ";
            }
            cout << endl;
        }
        
        evaluate_solution(s, &next_level);
        cout << "Projected solution fitness: " << s.fitness << endl;
        
        L++;
        edge_frequency.clear();
        
        cout << "⏱️  Level " << all_levels[L-1].level_id << " Tabu Search Time: " 
             << fixed << setprecision(10) << level_time << "s" << endl;
    }
    Solution best_overall = s;
    
    for (int i = 0; i < L; i++) {
        int current_level_id = L - i;
        int prev_level_id = L - i - 1;
        
        //cout << "\n=== REFINING FROM LEVEL " << current_level_id << " TO LEVEL " << prev_level_id << " ===" << endl; 
        // Unmerge solution
        auto unmerge_start = chrono::high_resolution_clock::now();
        s = unmerge_solution_to_previous_level(s, all_levels[current_level_id], all_levels[prev_level_id]);
        auto unmerge_end = chrono::high_resolution_clock::now();
        double unmerge_time = chrono::duration<double>(unmerge_end - unmerge_start).count();
        cout << "⏱️  Unmerging from level " << current_level_id << " to " << prev_level_id 
             << " Time: " << fixed << setprecision(10) << unmerge_time << "s" << endl;
        truck_times = all_levels[prev_level_id].truck_time_matrix;
        drone_times = all_levels[prev_level_id].drone_time_matrix;

        C1 = all_levels[prev_level_id].C1_level;
        C2 = all_levels[prev_level_id].C2_level;
        num_nodes = all_levels[prev_level_id].nodes.size();
        
        /*cout << "Level " << prev_level_id << " stats:" << endl;
        cout << "  Nodes: " << num_nodes << endl;
        cout << "  C1: " << C1.size() << ", C2: " << C2.size() << endl;
        cout << "  Matrix: " << distances.size() << "x" 
             << (distances.empty() ? 0 : distances[0].size()) << endl;*/

        update_node_index_cache(all_levels[prev_level_id]);
        // CASE 1: LEVEL 0 - DÙNG EVALUATE VÀ TABU KHÔNG CÓ LEVEL
        if (prev_level_id == 0) {
            cout << "\n🎯 FINAL REFINEMENT AT LEVEL 0 (No merged nodes)" << endl;
            merged_nodes_info.clear();
            
            // EVALUATE KHÔNG CÓ LEVEL (nullptr)
            evaluate_solution(s, nullptr);
            print_solution(s);
            
            auto refine_start = chrono::high_resolution_clock::now();
            edge_frequency.clear();
            
            s = tabu_search(s, nullptr, false);
            
            evaluate_solution(s, nullptr);
            
            auto refine_end = chrono::high_resolution_clock::now();
            double refine_time = chrono::duration<double>(refine_end - refine_start).count();
            cout << "⏱️  Final Refining at level 0 Time: " 
                << fixed << setprecision(10) << refine_time << "s" << endl;
            cout << "After tabu: " << endl;
            print_solution(s);
            best_overall = s;
        }
        // CASE 2: LEVEL 1, 2, 3...
        else {
            cout << "\n🔧 Refining at level " << prev_level_id << " (with merged nodes)" << endl;
            
            auto it = merged_nodes_info.begin();
            while (it != merged_nodes_info.end()) {
                if (it->second.level_id > prev_level_id) {
                    it = merged_nodes_info.erase(it);
                } else {
                    ++it;
                }
            }
            cout << "🧹 Cleaned merged_nodes_info: kept " << merged_nodes_info.size() 
                << " nodes for level " << prev_level_id << endl;
            
            evaluate_solution(s, &all_levels[prev_level_id]);
            print_solution(s);
            
            auto refine_start = chrono::high_resolution_clock::now();
            edge_frequency.clear();
            
            s = tabu_search(s, &all_levels[prev_level_id], false);
            evaluate_solution(s, &all_levels[prev_level_id]);
            
            auto refine_end = chrono::high_resolution_clock::now();
            double refine_time = chrono::duration<double>(refine_end - refine_start).count();
            cout << "⏱️  Refining at level " << prev_level_id << " Time: " 
                << fixed << setprecision(10) << refine_time << "s" << endl;
            cout << "After tabu: " << endl;
            print_solution(s);
            best_overall = s;
        }
    }

    return best_overall;
}

/*Solution create_test_solution_from_routes(const vector<vector<int>>& test_routes) {
    Solution test_sol;
    test_sol.route = test_routes;
    
    cout << "\n" << string(70, '=') << endl;
    cout << "🧪 TESTING WITH PREDEFINED ROUTES" << endl;
    cout << string(70, '=') << "\n" << endl;
    
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
    
    evaluate_solution(test_sol, nullptr);
    
    cout << "\n" << string(70, '=') << endl;
    cout << "📋 TEST RESULTS" << endl;
    cout << string(70, '=') << "\n" << endl;
    
    cout << "Makespan: " << test_sol.makespan << " min" << endl;
    cout << "Drone violation: " << test_sol.drone_violation << " min" << endl;
    cout << "Fitness: " << test_sol.fitness << endl;
    cout << "Is feasible: " << (test_sol.is_feasible ? "YES ✅" : "NO ❌") << endl;
    
    if (!test_sol.is_feasible) {
        cout << "\n⚠️  VIOLATIONS DETECTED:" << endl;
        
        if (test_sol.drone_violation > 0) {
            cout << "  🚁 Drone flight time exceeded by " << test_sol.drone_violation << " min" << endl;
            cout << "     → Some drones flew > " << vehicles[3].limit_drone << " min without returning to depot" << endl;
        }
        
    } else {
        cout << "\n ALL CONSTRAINTS SATISFIED!" << endl;
    }
    
    return test_sol;
}*/

void run_all_datasets() {
    vector<string> datasets = {
        "d:\\New folder\\instances\\C101_3.txt",
        "d:\\New folder\\instances\\C201_3.txt",
        "d:\\New folder\\instances\\R101_3.txt",
        "d:\\New folder\\instances\\RC101_3.txt"
    };
    
    for (const auto& dataset_path : datasets) {
        cout << "\n" << string(80, '=') << endl;
        cout << "📊 RUNNING: " << dataset_path << endl;
        cout << string(80, '=') << endl;
        
        read_dataset(dataset_path);

        cout << "\n=== CONFIGURATION ===" << endl;
        cout << "Nodes: " << num_nodes << " (1 depot + " << (num_nodes-1) << " customers)" << endl;
        cout << "C1: " << C1.size() << ", C2: " << C2.size() << endl;
        cout << "MAX_LEVELS: " << MAX_LEVELS << endl;
        cout << "MERGE_RATIO: " << (MERGE_RATIO * 100.0) << "%" << endl;
        cout << "MAX_ITER: " << MAX_ITER << endl;
        
        // Khởi tạo xe - Dynamic allocation based on problem size
        vehicles.clear();
        int customers = num_nodes - 1;
        int num_techs = 0, num_drones = 0;
        if (customers >= 6 && customers <= 12) {
            num_techs = 1;
            num_drones = 1;
        }
        else if (customers <= 20) {
            num_techs = 2;
            num_drones = 2;
        }
        else if (customers <= 50) {
            num_techs = 3;
            num_drones = 3;
        }
        else if (customers <= 100) {
            num_techs = 4;
            num_drones = 4;
        }
        else if (customers <= 200) {
            num_techs = 10;
            num_drones = 4;
        }
        else if (customers <= 500) {
            num_techs = 10;
            num_drones = 10;
        }
        else if (customers <= 1000) {
            num_techs = 15;
            num_drones = 15;
        }
        
        for (int i = 0; i < num_techs; ++i) {
            vehicles.push_back({ i, TRUCK_SPEED, false, 0.0 });
        }
        for (int i = 0; i < num_drones; ++i) {
            vehicles.push_back({ num_techs + i, DRONE_SPEED, true, 120.0 });
        }
        
        auto start_time = chrono::high_resolution_clock::now();
        Solution sol = multilevel_tabu_search();
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        
        cout << "\n📋 RESULT:" << endl;
        cout << "Makespan: " << sol.makespan << " min" << endl;
        cout << "Drone violation: " << sol.drone_violation << " min" << endl;
        cout << "Fitness: " << sol.fitness << endl;
        cout << "Feasible: " << (sol.is_feasible ? "✅ YES" : "❌ NO") << endl;
        cout << "Time: " << elapsed.count() << " seconds" << endl;
        
        cout << "\n✅ Finished dataset" << endl;
    }
}

int main(int argc, char* argv[]) {
    srand(time(nullptr));
    
    if (argc <= 1) {
        run_all_datasets();
        return 0;
    }

    string dataset_path = argv[1];

    if (argc > 2) {
        MAX_LEVELS = max(1, atoi(argv[2]));
        USE_MANUAL_SEGMENT_CONFIG = true;
    }

    if (argc > 5) {
        double ratio_arg = atof(argv[5]);
        if (ratio_arg > 1.0) {
            ratio_arg /= 100.0;
        }
        MERGE_RATIO = min(0.95, max(0.01, ratio_arg));
    }

    read_dataset(dataset_path);

    cout << "\n=== CONFIGURATION ===" << endl;
    cout << "MAX_LEVELS: " << MAX_LEVELS << endl;
    cout << "MERGE_RATIO: " << (MERGE_RATIO * 100.0) << "%" << endl;
    cout << "MAX_ITER (= iter/segment * segments/level): " << MAX_ITER << endl;

    printf("MAX_ITER: %d\n", MAX_ITER);
 
    // Khởi tạo danh sách xe 
    vehicles.clear();
    int customers = num_nodes-1;
    int num_techs = 0, num_drones = 0;
    if (customers >= 6 && customers <= 12) {
        num_techs = 1;
        num_drones = 1;
    }
    else if (customers <= 20) {
        num_techs = 2;
        num_drones = 2;
    }
    else if (customers <= 50) {
        num_techs = 3;
        num_drones = 3;
    }
    else if (customers <= 100) {
        num_techs = 4;
        num_drones = 4;
    }
    else if (customers <= 200) {
        num_techs = 10;
        num_drones = 4;
    }
    else if (customers <= 500) {
        num_techs = 10;
        num_drones = 10;
    }
    else if (customers <= 1000) {
        num_techs = 15;
        num_drones = 15;
    }

    for (int i = 0; i < num_techs; ++i) {
        vehicles.push_back({ i+1, 0.58f, false, 0.0f }); // technician
    }
    for (int i = 0; i < num_drones; ++i) {
        vehicles.push_back({ num_techs + i + 1, 0.83f, true, 120.0f }); // drone
    }

    /*vector<vector<int>> test_routes = {
        // 3 Technicians
        {0, 43, 49, 48, 15, 44, 34, 0},
        {0, 30, 9, 16, 23, 12, 0},
        {0, 8, 26, 1, 11, 38, 4, 14, 32, 0},
        
        // 3 Drones
        {0, 37, 41, 40, 25, 42, 21, 13, 47, 31, 0},
        {0, 19, 3, 18, 45, 22, 29, 5, 10, 33, 46, 17, 0, 24, 0},
        {0, 2, 6, 28, 35, 20, 27, 39, 7, 36, 50, 0}
    };*/

    //Solution test_solution = create_test_solution_from_routes(test_routes);

    Solution best_solution = multilevel_tabu_search();
    print_solution(best_solution);

    return 0;
}