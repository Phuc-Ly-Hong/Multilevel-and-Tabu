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

struct MultilevelSolution {
    vector<Solution> level_solutions; // lời giải ở các cấp độ
};

vector<vector<double>> distances;
vector<Node> C1; // customers served only by technicians
vector<Node> C2; // customers served by drones or technicians
vector<VehicleFamily> vehicles;

int depot_id = 0;
int num_nodes = 0;
double alpha1 = 1.0; // tham số hàm phạt thứ nhất
double alpha2 = 1.0; // tham số hàm phạt thứ hai
double Beta = 0.5; // tham số điều chỉnh hệ số hàm phạt

int MAX_ITER;
int TABU_TENURE;
int MAX_NO_IMPROVE = 10000;
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
        MAX_ITER = 300 * nodes.size() / 2;
        SEGMENT_LENGTH = 800;
    } else if (nodes.size() > 50){
        MAX_ITER = 12000;
        SEGMENT_LENGTH = 800;
    } else {
        MAX_ITER = 12000;
        SEGMENT_LENGTH = 500;
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

void evaluate_solution(Solution &sol){
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
                    current_time += distances[prev][depot_id] / vehicles[i].speed;
                }
                double arrival_depot = current_time;
                double flight_time = arrival_depot - depart_time;
                if (vehicles[i].is_drone && flight_time > vehicles[i].limit_drone){
                    sol.drone_violation += (flight_time - vehicles[i].limit_drone);
                }
                for (auto &p : served_in_trip){
                    double time_served = p.second;
                    double wait_time = arrival_depot - time_served;
                    if (wait_time > C2[0].limit_wait) {
                        sol.waiting_violation += (wait_time - C2[0].limit_wait);
                    }
                }
                if (sol.drone_violation >0 || sol.waiting_violation >0) sol.is_feasible = false;
                depart_time = current_time;
                served_in_trip.clear();
                prev = depot_id;
            }
            else{
                current_time += distances[prev][cid] / vehicles[i].speed;
                served_in_trip.push_back({cid, current_time}); // thời điểm khách hàng được phục vụ
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

    // Khởi tạo: mỗi xe bắt đầu từ depot
    for (size_t v = 0; v < vehicles.size(); ++v)
        sol.route[v].push_back(depot_id);

    /*// --- Gán C1 cho technician trước ---
    vector<int> unserved_C1;
    for (const auto& n : C1) unserved_C1.push_back(n.id);

    // Vị trí hiện tại của từng technician
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
    } */

    // --- Gán C2 cho tất cả xe ---
    vector<int> unserved_C2;
    for (const auto& n : C2) unserved_C2.push_back(n.id);

    // Vị trí hiện tại của từng xe
    vector<int> current_pos(vehicles.size(), depot_id);
    for (size_t v = 0; v < vehicles.size(); v++) {
        if (sol.route[v].size() > 1) {
            current_pos[v] = sol.route[v][sol.route[v].size() - 1];
        }
    }

    while (!unserved_C2.empty()) {
        for (size_t v = 0; v < vehicles.size(); v++) {
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
            }
        }
    }

    // Kết thúc: đảm bảo mỗi route kết thúc bằng depot
    for (size_t v = 0; v < vehicles.size(); v++) {
        if (sol.route[v].empty() || sol.route[v].back() != depot_id) {
            sol.route[v].push_back(depot_id);
        }
    }

    evaluate_solution(sol);
    return sol;
}

int get_type(int nid) {
    //for (const auto& n : C1) if (n.id == nid) return 1;
    for (const auto& n : C2) if (n.id == nid) return 2;
    return -1;
}

RouteAnalysis analyze_drone_route(const vector<int> &route, int vehicle_idx) {
    RouteAnalysis analysis;
    
    // ← XÓA check is_drone, vì cần phân tích cả tech
    if (route.size() <= 2) {
        return analysis;
    }
    
    analysis.cumulative_flight_time.resize(route.size(), 0.0);
    analysis.arrival_time.resize(route.size(), 0.0);
    analysis.waiting_times.resize(route.size(), 0.0);
    
    double current_time = 0.0;
    double flight_time_since_last_depot = 0.0;
    double max_flight_segment = 0.0;
    int last_pos = depot_id;
    
    vector<pair<int, double>> current_trip_customers;
    double trip_start_time = 0.0;
    
    for (size_t i = 0; i < route.size(); i++) {
        int current_node = route[i];
        
        if (i > 0) {
            double travel_time = distances[last_pos][current_node] / vehicles[vehicle_idx].speed;
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
        last_pos = current_node;
    }
    
    analysis.total_flight_time = max_flight_segment;
    
    return analysis;
}

// Tìm vị trí tốt nhất để chèn depot vào route
int find_best_depot_insertion(const vector<int> &route, int vehicle_idx) {
    // Chỉ xử lý drone và route đủ dài
    if (!vehicles[vehicle_idx].is_drone || route.size() <= 3) {
        return -1;
    }
    
    // Phân tích route hiện tại
    RouteAnalysis original = analyze_drone_route(route, vehicle_idx);
    
    // Kiểm tra xem có vi phạm không
    double original_violation = 0.0;
    if (original.total_flight_time > vehicles[vehicle_idx].limit_drone) {
        original_violation += alpha1 * (original.total_flight_time - vehicles[vehicle_idx].limit_drone);
    }
    original_violation += alpha2 * original.total_waiting;
    
    // Nếu không vi phạm gì thì không cần chèn
    if (original_violation < EPSILON) {
        return -1;
    }
    
    double best_improvement = 0.0;
    int best_pos = -1;
    
    // Thử chèn depot vào từng vị trí
    for (size_t pos = 2; pos < route.size() - 1; pos++) {
        // Không chèn depot liên tiếp
        if (route[pos - 1] == depot_id || route[pos] == depot_id) {
            continue;
        }
        
        // Tạo route thử nghiệm với depot chèn vào
        vector<int> test_route = route;
        test_route.insert(test_route.begin() + pos, depot_id);
        
        // Phân tích route mới
        RouteAnalysis test_analysis = analyze_drone_route(test_route, vehicle_idx);
        
        // Tính vi phạm của route mới
        double test_violation = 0.0;
        if (test_analysis.total_flight_time > vehicles[vehicle_idx].limit_drone) {
            test_violation += alpha1 * (test_analysis.total_flight_time - vehicles[vehicle_idx].limit_drone);
        }
        test_violation += alpha2 * test_analysis.total_waiting;
        
        // Tính improvement (giảm bao nhiêu vi phạm)
        double improvement = original_violation - test_violation;
        
        // Penalty nhỏ cho việc tăng quãng đường (vì phải về depot)
        double detour_distance = distances[route[pos - 1]][depot_id] + 
                                 distances[depot_id][route[pos]] - 
                                 distances[route[pos - 1]][route[pos]];
        double detour_penalty = 0.05 * detour_distance; // penalty nhỏ 5%
        
        improvement -= detour_penalty;
        
        // Lưu vị trí tốt nhất
        if (improvement > best_improvement) {
            best_improvement = improvement;
            best_pos = pos;
        }
    }
    
    // Chỉ chèn nếu cải thiện đáng kể (> 0.5)
    return (best_improvement > 0.5) ? best_pos : -1;
}

void optimize_all_drone_routes(Solution &sol) {
    bool changed = true;
    int max_rounds = 3; // Tối đa 3 vòng chèn
    int round = 0;
    
    cout << "\n=== OPTIMIZING DRONE ROUTES ===" << endl;
    
    while (changed && round < max_rounds) {
        changed = false;
        round++;
        cout << "Round " << round << ":" << endl;
        
        for (size_t v = 0; v < vehicles.size(); v++) {
            if (!vehicles[v].is_drone) continue;
            
            int insert_pos = find_best_depot_insertion(sol.route[v], v);
            
            if (insert_pos != -1) {
                // Lưu trạng thái trước khi chèn
                double old_fitness = sol.fitness;
                
                // Chèn depot
                sol.route[v].insert(sol.route[v].begin() + insert_pos, depot_id);
                evaluate_solution(sol);
                
                changed = true;
                
                cout << "  Vehicle " << v << ": Inserted depot at position " << insert_pos 
                     << " (fitness: " << old_fitness << " -> " << sol.fitness << ")" << endl;
            }
        }
        
        if (!changed) {
            cout << "  No more beneficial insertions found." << endl;
        }
    }
    
    cout << "=== OPTIMIZATION COMPLETE ===" << endl;
    cout << "Final fitness: " << sol.fitness << endl;
}

void remove_redundant_depots(Solution &sol) {
    bool changed = true;
    int round = 0;
    
    cout << "\n=== REMOVING REDUNDANT DEPOTS ===" << endl;
    
    while (changed && round < 5) {
        changed = false;
        round++;
        cout << "Round " << round << ":" << endl;
        
        for (size_t v = 0; v < vehicles.size(); v++) {
            if (!vehicles[v].is_drone) continue;
            
            vector<int> &route = sol.route[v];
            
            // Tìm các depot trung gian (không phải depot đầu/cuối)
            for (size_t i = 1; i < route.size() - 1; ) {
                if (route[i] == depot_id) {
                    // Lưu trạng thái trước khi xóa
                    double old_fitness = sol.fitness;
                    
                    // Thử xóa depot này
                    vector<int> test_route = route;
                    test_route.erase(test_route.begin() + i);
                    
                    Solution test_sol = sol;
                    test_sol.route[v] = test_route;
                    evaluate_solution(test_sol);
                    
                    // Nếu xóa mà fitness KHÔNG TỆ HƠN (hoặc tốt hơn)
                    if (test_sol.fitness <= sol.fitness + EPSILON) {
                        route = test_route;
                        sol = test_sol;
                        changed = true;
                        
                        cout << "  Vehicle " << v << ": Removed depot at position " << i 
                             << " (fitness: " << old_fitness << " -> " << sol.fitness << ")" << endl;
                        
                        // Không tăng i vì đã xóa phần tử
                    } else {
                        i++; // Giữ depot này, chuyển sang vị trí tiếp theo
                    }
                } else {
                    i++;
                }
            }
        }
        
        if (!changed) {
            cout << "  No more depots to remove." << endl;
        }
    }
    
    cout << "=== REMOVAL COMPLETE ===" << endl;
    cout << "Final fitness: " << sol.fitness << endl;
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

Solution move_1_0(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
    Solution new_sol = current_sol;
    int cid = new_sol.route[v1][pos1];
    if (cid == depot_id) return current_sol; // không di chuyển depot
        // Kiểm tra không được chèn vào vị trí 0 (trước depot xuất phát)
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

    evaluate_solution(new_sol);
    return new_sol;
}

Solution move_1_1(Solution current_sol, size_t v1, size_t node1, size_t v2, size_t node2){
    Solution new_sol = current_sol;
    swap(new_sol.route[v1][node1], new_sol.route[v2][node2]);
    evaluate_solution(new_sol);
    return new_sol;
}

Solution move_2_0(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
    Solution new_sol = current_sol;
    int cid1 = new_sol.route[v1][pos1];
    int cid2 = new_sol.route[v1][pos1+1];
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
    new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid1);
    new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2 + 1, cid2);
    evaluate_solution(new_sol);
    return new_sol;
}

Solution move_2_1(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
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
    
    evaluate_solution(new_sol);
    return new_sol;
}

Solution move_2_2(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
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
    
    evaluate_solution(new_sol);
    return new_sol;
}

Solution move_2opt(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
    Solution new_sol = current_sol;
    
    // same trip
    if (v1 == v2) {
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
    // different trip
    else {
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
    
    evaluate_solution(new_sol);
    return new_sol;
}

Solution tabu_search(){
    Solution initial_sol = init_greedy_solution();
    optimize_all_drone_routes(initial_sol);
    remove_redundant_depots(initial_sol);

    Solution best_sol = initial_sol;
    Solution current_sol = initial_sol;

    vector<TabuMove> tabu_list; // danh sách các move bị tabu
    int no_improve_count = 0;
    int last_depot_opt_iter = 0;

    int no_improve_segment_length = 0;
    const int max_no_improve_segment = 8;

    vector<string> move_types = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
    
    for (int iter = 0; iter < MAX_ITER && no_improve_count < MAX_NO_IMPROVE; iter++){
        bool should_depot_opt = false;
        if (iter > 0 && iter % 100 == 0) {
            should_depot_opt = true;
        }
        if (no_improve_count >= 50 && (iter - last_depot_opt_iter) >=25) {
            should_depot_opt = true;
        }
        if (current_sol.drone_violation > 10.0 || current_sol.waiting_violation > 100.0) {
            if (iter - last_depot_opt_iter >= 20) {
                should_depot_opt = true;
            }
        }
        
        if (should_depot_opt) {
            cout << "\n=== DEPOT OPTIMIZATION at iter " << iter 
                 << " (reason: " << (iter % 100 == 0 ? "periodic" : 
                                   no_improve_count >= 50 ? "stuck" : "high_violation") 
                 << ") ===" << endl;
                 
            Solution temp_sol = current_sol;
            double old_fitness = temp_sol.fitness;
            
            optimize_all_drone_routes(temp_sol);
            remove_redundant_depots(temp_sol);
            
            if (temp_sol.fitness < current_sol.fitness - EPSILON) {
                current_sol = temp_sol;
                last_depot_opt_iter = iter;
                
                cout << "  Depot opt success: " << old_fitness << " -> " << current_sol.fitness << endl;
                
                if (current_sol.fitness < best_sol.fitness - EPSILON) {
                    best_sol = current_sol;
                    no_improve_count = 0;
                    cout << "  ✅ NEW BEST: " << best_sol.fitness << endl;
                }
            }
        }

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

                    for (size_t v2 = 0; v2 < current_sol.route.size(); v2++) {
                        if (v1 == v2) continue;
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++) {
                            if (get_type(n1) == 1 && (vehicles[v2].is_drone || vehicles[v1].is_drone)) continue; // C1 không thể giao cho drone

                            Solution new_sol = move_1_0(current_sol, v1, pos1, v2, pos2);
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
                            if (n2 == depot_id || n1 == n2 || get_type(n1) != get_type(n2) || ((abs(int(pos1)-int(pos2)) <= 1) && (v1 == v2))) continue;

                            Solution new_sol = move_1_1(current_sol, v1, pos1, v2, pos2);
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
                        if ((get_type(n1) == 1 || get_type(n2) == 1) && vehicles[v2].is_drone) continue;
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++){

                            Solution new_sol = move_2_0(current_sol, v1, pos1, v2, pos2);
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
                            if ((get_type(n1) == 1 || get_type(n2) == 1) && vehicles[v2].is_drone) continue;
                            if (get_type(n3) == 1 && vehicles[v1].is_drone) continue;
                            Solution new_sol = move_2_1(current_sol, v1, pos1, v2, pos2);
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
                            if ((get_type(n1) == 1 || get_type(n2) == 1) && vehicles[v2].is_drone) continue;
                            if ((get_type(n3) == 1 || get_type(n4) == 1) && vehicles[v1].is_drone) continue;

                            Solution new_sol = move_2_2(current_sol, v1, pos1, v2, pos2);
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
                        
                        Solution new_sol = move_2opt(current_sol, v1, pos1, v1, pos2); // Cùng xe v1
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
                                    if (cid != depot_id && get_type(cid) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (!invalid_move && vehicles[v1].is_drone) {
                                for (size_t i = pos2; i < current_sol.route[v2].size() - 1; i++) {
                                    int cid = current_sol.route[v2][i];
                                    if (cid != depot_id && get_type(cid) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (invalid_move) continue;

                            int customer_at_pos1 = current_sol.route[v1][pos1];
                            int customer_at_pos2 = current_sol.route[v2][pos2];
                            
                            Solution new_sol = move_2opt(current_sol, v1, pos1, v2, pos2); 
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
            evaluate_solution(current_sol);

            cout << "Iter: " << iter << " Move: " << move_type 
                 << " current makespan: " << current_sol.makespan 
                 << ", drone_violation: " << current_sol.drone_violation 
                 << ", waiting_violation: " << current_sol.waiting_violation 
                 << ", fitness: " << current_sol.fitness << endl;
            cout << "Route details:" << endl;
            for (size_t v = 0; v < current_sol.route.size(); v++) {
                cout << "Vehicle " << v << ": ";
                for (int cid : current_sol.route[v]) cout << cid << " ";
                cout << endl;
            }

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
            cout << "Tabu move added: type=" << best_move.type
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
                 << ", tenure=" << best_move.tenure << endl;
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

            cout << "Tabu move added: type=" << reverse_move.type
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
                 << ", tenure=" << reverse_move.tenure << endl;
            
            if (current_sol.is_feasible && current_sol.fitness < best_sol.fitness - EPSILON){
                best_sol = current_sol;
                no_improve_count = 0;
                scorePi[move_type_idx] += delta1;
                segment_improved = true;
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
            cout << "SEGMENT " << (iter + 1)/SEGMENT_LENGTH << " COMPLETE" << endl;
            cout << "No improve segments: " << no_improve_segment_length <<"/"<< max_no_improve_segment << endl;
            cout << "Updated weights: ";
            for (size_t i = 0; i < MOVE_SET.size(); i++) {
                cout << MOVE_SET[i] << "=" << weights[i] << " ";
            }
            cout << endl;
            cout << "Current best fitness: " << best_sol.fitness << endl;
        }
    }
    optimize_all_drone_routes(best_sol);
    remove_redundant_depots(best_sol);
    return best_sol;
}

int main(){
    srand(time(nullptr));
    read_dataset("D:\\New folder\\instances\\100.40.1.txt");
    printf(" %d\n", MAX_ITER);
 
    // Khởi tạo danh sách xe 
    vehicles.clear();
    int customers = num_nodes-1;
    int pairs = 0;
    if (customers >= 6 && customers <= 12) pairs = 1;
    else if (customers <= 20) pairs = 2;
    else if (customers <= 50) pairs = 3;
    else if (customers <= 100) pairs = 4;
    for (int i = 0; i < pairs; ++i) {
        vehicles.push_back({ i+1, 0.58f, false, 0.0 }); // technician
    }
    for (int i = 0; i < pairs; ++i) {
        vehicles.push_back({ pairs + i + 1, 0.83f, true, 120.0 }); // drone
    }

    Solution sol = tabu_search();
    print_solution(sol);

    return 0;
}



