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
    vector<double> route_time; // thời gian hoàn thành theo từng xe
    vector<double> route_drone_violation; // vi phạm theo từng xe

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

struct RouteEval {
    double time;
    double drone_violation;
};

vector<vector<double>> distances;
vector<vector<double>> truck_times;
vector<vector<double>> drone_times;
vector<Node> C1; // customers served only by technicians
vector<Node> C2; // customers served by drones or technicians
vector<VehicleFamily> vehicles;
vector<int> node_type; // 1 = C1, 2 = C2

constexpr double TRUCK_SPEED = 0.58;
constexpr double DRONE_SPEED = 0.83;

int depot_id = 0;
int num_nodes = 0;
double alpha1 = 1.0; // tham số hàm phạt thứ nhất
double alpha2 = 1.0; // tham số hàm phạt thứ hai
double Beta = 0.5; // tham số điều chỉnh hệ số hàm phạt

int MAX_ITER;
int TABU_TENURE;
double EPSILON = 1e-6;
// Match Multilevel_Tabu_no_lwt: up to (2 * depth + 1) Tabu calls.
int MATCHED_MAX_LEVELS = 5;
double ITER_K = 15;
long long ITERATIONS_USED = 0;
double TABU_TENURE_FACTOR = 0.25;
double tabu1 = 0.07;
double tabu2 = 0.11;

// Adaptive parameters
vector<string> MOVE_SET = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
vector<double> scorePi = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
vector<double> used_count = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

double delta1 = 0.8;
double delta2 = 0.6;
double delta3 = 0.1;
double delta4 = 0.1;

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
    for (const auto& node : nodes) {
        if (node.id == depot_id) {
            cout << "Node id: " << node.id << " (depot), x: " << node.x << ", y: " << node.y << endl;
            continue;
        } else {
            cout << "Node id: " << node.id << ", x: " << node.x << ", y: " << node.y
                 << ", type: " << (node.c1_or_c2 > 0 ? "C2" : "C1") << endl;
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

    build_time_matrices_from_distance(distances, truck_times, drone_times);

    // Phân loại khách hàng
    node_type.assign(nodes.size(), -1);
    for (const auto& node : nodes){
        if (node.id == depot_id) continue;
        if (node.c1_or_c2 > 0){
            C2.push_back(node);
            if (node.id >= 0 && node.id < (int)node_type.size()) node_type[node.id] = 2;
        } else if (node.c1_or_c2 == 0) {
            C1.push_back(node);
            if (node.id >= 0 && node.id < (int)node_type.size()) node_type[node.id] = 1;
        }
    }
    cout << "C1 size: " << C1.size() << ", C2 size: " << C2.size() << endl;
    num_nodes = nodes.size();
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

void normalize_route(vector<int> &route);

void print_drone_violation_details(const Solution &sol){
    bool any = false;
    cout << "Drone violation details:" << endl;
    for (size_t v = 0; v < sol.route.size(); v++) {
        if (!vehicles[v].is_drone) continue;
        vector<int> route = sol.route[v];
        normalize_route(route);
        const vector<vector<double>>& time_matrix = drone_times;
        int prev = depot_id;
        double current_time = 0.0;
        double depart_time = 0.0;
        int trip_idx = 0;

        for (int cid : route) {
            if (cid == depot_id) {
                if (prev != depot_id) current_time += time_matrix[prev][depot_id];
                double arrival_depot = current_time;
                double flight_time = arrival_depot - depart_time;
                if (flight_time > vehicles[v].limit_drone + EPSILON) {
                    any = true;
                    cout << "Vehicle " << v
                         << " trip " << trip_idx
                         << " flight=" << flight_time
                         << " limit=" << vehicles[v].limit_drone
                         << " violation=" << (flight_time - vehicles[v].limit_drone)
                         << endl;
                }
                depart_time = current_time;
                prev = depot_id;
                trip_idx++;
            } else {
                current_time += time_matrix[prev][cid];
                prev = cid;
            }
        }
    }
    if (!any) cout << "No drone violations." << endl;
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

RouteEval evaluate_route(vector<int> &route, const VehicleFamily &vehicle) {
    normalize_route(route);
    const vector<vector<double>>& time_matrix = vehicle.is_drone ? drone_times : truck_times;
    int prev = depot_id;
    double current_time = 0.0;
    double depart_time = 0.0;
    double violation = 0.0;

    for (int cid : route) {
        if (cid == depot_id) {
            if (prev != depot_id) current_time += time_matrix[prev][depot_id];
            double arrival_depot = current_time;
            double flight_time = arrival_depot - depart_time;
            if (vehicle.is_drone && flight_time > vehicle.limit_drone) {
                violation += (flight_time - vehicle.limit_drone);
            }
            depart_time = current_time;
            prev = depot_id;
        } else {
            current_time += time_matrix[prev][cid];
            prev = cid;
        }
    }

    return {current_time, violation};
}

void evaluate_solution(Solution &sol) {
    for (auto &route : sol.route) normalize_route(route);

    sol.makespan = 0;
    sol.drone_violation = 0;
    sol.fitness = 0;
    sol.is_feasible = true;

    sol.route_time.assign(sol.route.size(), 0.0);
    sol.route_drone_violation.assign(sol.route.size(), 0.0);

    for (size_t i = 0; i < sol.route.size(); i++){
        RouteEval eval = evaluate_route(sol.route[i], vehicles[i]);
        sol.route_time[i] = eval.time;
        sol.route_drone_violation[i] = eval.drone_violation;
        sol.makespan = max(sol.makespan, eval.time);
        sol.drone_violation += eval.drone_violation;
    }

    if (sol.drone_violation > 0) sol.is_feasible = false;

    sol.fitness = sol.makespan + alpha1*sol.drone_violation; // + alpha2*sol.waiting_violation; // Removed waiting violation
}

void recompute_solution_from_cache(Solution &sol) {
    sol.makespan = 0.0;
    sol.drone_violation = 0.0;
    for (size_t i = 0; i < sol.route_time.size(); i++) {
        sol.makespan = max(sol.makespan, sol.route_time[i]);
        sol.drone_violation += sol.route_drone_violation[i];
    }
    sol.is_feasible = (sol.drone_violation <= EPSILON);
    sol.fitness = sol.makespan + alpha1 * sol.drone_violation;
}

void recompute_solution_for_routes(Solution &sol, size_t v1, size_t v2, bool has_second) {
    if (sol.route_time.size() != sol.route.size() || sol.route_drone_violation.size() != sol.route.size()) {
        evaluate_solution(sol);
        return;
    }
    RouteEval eval1 = evaluate_route(sol.route[v1], vehicles[v1]);
    sol.route_time[v1] = eval1.time;
    sol.route_drone_violation[v1] = eval1.drone_violation;
    if (has_second && v2 != v1) {
        RouteEval eval2 = evaluate_route(sol.route[v2], vehicles[v2]);
        sol.route_time[v2] = eval2.time;
        sol.route_drone_violation[v2] = eval2.drone_violation;
    }
    recompute_solution_from_cache(sol);
}

void recompute_solution_for_route(Solution &sol, size_t v1) {
    recompute_solution_for_routes(sol, v1, v1, false);
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
            double best_time = DBL_MAX;
            int best_idx = -1;
            const vector<vector<double>>& time_matrix = vehicles[v].is_drone ? drone_times : truck_times;
            for (size_t i = 0; i < unserved_C2.size(); i++) {
                int cid = unserved_C2[i];
                double t = time_matrix[current_pos[v]][cid];
                if (t < best_time) {
                    best_time = t;
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

bool contains_depot_in_range(const vector<int>& route, size_t start, size_t end) {
    for (size_t i = start; i <= end && i < route.size(); i++) {
        if (route[i] == depot_id) return true;
    }
    return false;
}

int get_type(int nid) {
    if (nid >= 0 && nid < (int)node_type.size()) return node_type[nid];
    return -1;
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

    if (pos1 == 0 || pos1 == new_sol.route[v1].size() - 1) {
        return current_sol;
    }

    if (pos2 == new_sol.route[v2].size() && vehicles[v2].is_drone){
        if (get_type(cid) == 1) return current_sol; 
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
        if (get_type(cid) == 1 && vehicles[v2].is_drone) {
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid);
    }
    
    recompute_solution_for_routes(new_sol, v1, v2, true);
    return new_sol;
}

Solution move_1_1(Solution current_sol, size_t v1, size_t node1, size_t v2, size_t node2){
    Solution new_sol = current_sol;
    int cid1 = new_sol.route[v1][node1];
    int cid2 = new_sol.route[v2][node2];
    if (cid1 == depot_id || cid2 == depot_id) return current_sol; // không di chuyển depot
    swap(new_sol.route[v1][node1], new_sol.route[v2][node2]);
    recompute_solution_for_routes(new_sol, v1, v2, true);
    return new_sol;
}

Solution move_2_0(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
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
        if (get_type(cid1) == 1 || get_type(cid2) == 1){
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].push_back(cid1);
        new_sol.route[v2].push_back(cid2);
        new_sol.route[v2].push_back(depot_id);
    } else {
        if ((get_type(cid1) == 1 || get_type(cid2) == 1) && vehicles[v2].is_drone){
            return current_sol;
        }
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1 + 1);
        new_sol.route[v1].erase(new_sol.route[v1].begin() + pos1);
        new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2, cid1);
        new_sol.route[v2].insert(new_sol.route[v2].begin() + pos2 + 1, cid2);
    }

    recompute_solution_for_routes(new_sol, v1, v2, true);
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

    recompute_solution_for_routes(new_sol, v1, v2, true);
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

    recompute_solution_for_routes(new_sol, v1, v2, true);
    return new_sol;
}

Solution move_2opt(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2){
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

    if (v1 == v2) {
        recompute_solution_for_route(new_sol, v1);
    } else {
        recompute_solution_for_routes(new_sol, v1, v2, true);
    }
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

Solution tabu_search(){
    const int num_customer = max(1, num_nodes - 1);
    const int min_tenure = max(1, (int)ceil(tabu1 * num_customer));
    const int max_tenure = max(min_tenure, (int)ceil(tabu2 * num_customer));
    TABU_TENURE = max(min_tenure, min((int)ceil(num_customer * TABU_TENURE_FACTOR), max_tenure));
    ITERATIONS_USED = 0;
    Solution initial_sol = init_greedy_solution();
    Solution best_sol = initial_sol;
    Solution current_sol = initial_sol;

    vector<TabuMove> tabu_list; // danh sách các move bị tabu
    int last_depot_opt_iter = 0;

    vector<string> move_types = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
    
    for (int iter = 0; iter < MAX_ITER; iter++){
        ++ITERATIONS_USED;
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
                                if (get_type(n1) == 1) continue;
                                if (v1 == v2) continue;
                            } else {
                                if (v1 == v2) continue;
                                if (pos2 == current_sol.route[v2].size() - 1) continue;
                                if (get_type(n1) == 1 && vehicles[v2].is_drone) continue;
                            }

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
                        for (size_t pos2 = 1; pos2 <= current_sol.route[v2].size(); pos2++){
                            if (pos2 == current_sol.route[v2].size() && !vehicles[v2].is_drone) {
                                continue;
                            }

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
                            
                            Solution new_sol = move_2opt(current_sol, v1, pos1, v2, pos2); // Khác xe v1 và v2
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
                scorePi[move_type_idx] += delta1;
                segment_improved = true;
            } else if (current_sol.fitness < current_fitness - EPSILON) {
                scorePi[move_type_idx] += delta2;
            } else {
                scorePi[move_type_idx] += delta3;
            }
        }

        update_weights();
    }
    return best_sol;
}

int main(int argc, char* argv[]){
    try {
    string dataset_path = "D:\\New folder\\instances\\50.10.1.txt";
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    long long budget_override = -1;
    int next_arg = 1;
    if (next_arg < argc && string(argv[next_arg]).rfind("--", 0) != 0)
        dataset_path = argv[next_arg++];
    // Preserve the legacy positional total-iteration override.
    if (next_arg < argc && string(argv[next_arg]).rfind("--", 0) != 0) {
        long long legacy_budget = stoll(argv[next_arg++]);
        if (legacy_budget > 0) budget_override = legacy_budget;
    }
    for (int i = next_arg; i < argc; ++i) {
        string key = argv[i];
        if (key == "--help") {
            cout << "Tabu_no_lwt [instance.txt] [total_iterations] [options]\n"
                 << "--instance FILE --seed S --iter_k K --max_levels L\n"
                 << "--budget B (alias --max_iter): total Tabu iterations\n"
                 << "--tabu1 F --tabu2 F --tabu_factor F --delta1 F --delta2 F --delta3 F --delta4 F\n"
                 << "Defaults match elite 3549: K=15, matched ML depth=5; budget=165*customers.\n"
                 << "Use the same explicit --seed for this program and Multilevel_Tabu_no_lwt.\n";
            return 0;
        }
        if (i + 1 >= argc) throw invalid_argument("Missing value for " + key);
        string val = argv[++i];
        if (key == "--instance") dataset_path = val;
        else if (key == "--seed") seed = static_cast<unsigned int>(stoul(val));
        else if (key == "--iter_k") ITER_K = stod(val);
        else if (key == "--max_levels") MATCHED_MAX_LEVELS = stoi(val);
        else if (key == "--budget" || key == "--max_iter") budget_override = stoll(val);
        else if (key == "--tabu1") tabu1 = stod(val);
        else if (key == "--tabu2" || key == "--tabu_cap") tabu2 = stod(val);
        else if (key == "--tabu_factor") TABU_TENURE_FACTOR = stod(val);
        else if (key == "--delta1") delta1 = stod(val);
        else if (key == "--delta2") delta2 = stod(val);
        else if (key == "--delta3") delta3 = stod(val);
        else if (key == "--delta4") delta4 = stod(val);
        else throw invalid_argument("Unknown option: " + key);
    }
    if (!isfinite(ITER_K) || ITER_K <= 0 || MATCHED_MAX_LEVELS < 1 || MATCHED_MAX_LEVELS > 100
        || budget_override < -1)
        throw invalid_argument("Require iter_k > 0, 1 <= max_levels <= 100 and budget >= 0");
    if (!isfinite(tabu1) || !isfinite(tabu2) || tabu1 <= 0 || tabu2 < tabu1)
        throw invalid_argument("Require 0 < tabu1 <= tabu2");
    if (!isfinite(TABU_TENURE_FACTOR) || TABU_TENURE_FACTOR < 0)
        throw invalid_argument("Invalid tabu_factor");
    for (double delta : {delta1, delta2, delta3, delta4})
        if (!isfinite(delta) || delta < 0 || delta > 1) throw invalid_argument("Invalid delta");
    srand(seed);
    read_dataset(dataset_path);
    double phase_value = ITER_K * (num_nodes - 1);
    if (!isfinite(phase_value) || phase_value > numeric_limits<int>::max())
        throw invalid_argument("Phase iteration count is too large");
    long long phase_iterations = max(1LL, llround(phase_value));
    long long total_budget = budget_override >= 0 ? budget_override
        : phase_iterations * (2LL * MATCHED_MAX_LEVELS + 1);
    if (total_budget > numeric_limits<int>::max()) throw invalid_argument("Total iteration budget is too large");
    MAX_ITER = static_cast<int>(total_budget);
    cout << "\n=== CONFIGURATION ===\nMAX_ITER: " << MAX_ITER << '\n';
    cout << "CONFIG seed=" << seed << " iter_k=" << ITER_K
         << " matched_max_levels=" << MATCHED_MAX_LEVELS << " phase_max_iter=" << phase_iterations
         << " iteration_budget=" << MAX_ITER << " tabu1=" << tabu1 << " tabu2=" << tabu2
         << " delta1=" << delta1 << " delta2=" << delta2 << " delta3=" << delta3 << " delta4=" << delta4 << '\n';

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
        num_techs = 2;
        num_drones = 2;
    }
    else if (customers <= 100) {
        num_techs = 3;
        num_drones = 3;
    }
    else if (customers <= 200) {
        num_techs = 5;
        num_drones = 5;
    }
    else if (customers <= 500) {
        num_techs = 9;
        num_drones = 9;
    }
    else if (customers <= 1000) {
        num_techs = 15;
        num_drones = 15;
    }

    for (int i = 0; i < num_techs; ++i) {
        vehicles.push_back({ i+1, 0.58f, false, 0.0f }); // technician
    }
    for (int i = 0; i < num_drones; ++i) {
        vehicles.push_back({ num_techs + i + 1, 0.83f, true, 60.0f }); // drone
    }

    Solution sol = tabu_search();
    print_solution(sol);
    cout << "SEARCH iterations_used=" << ITERATIONS_USED << " tabu_tenure=" << TABU_TENURE << " tabu_calls=1" << endl;
    if (sol.drone_violation > EPSILON) {
        print_drone_violation_details(sol);
    }

    return 0;
    } catch (const exception& exc) {
        cerr << "Error: " << exc.what() << endl;
        return 1;
    }
}
