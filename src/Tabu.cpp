#include<bits/stdc++.h>
using namespace std;

enum MoveType : int { MT_1_0=0, MT_1_1=1, MT_2_0=2, MT_2_1=3, MT_2_2=4, MT_2OPT=5, MT_NONE=6 };
static const char* MT_NAME[] = {"1-0","1-1","2-0","2-1","2-2","2-opt","none"};

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
    vector<double> route_time; // thời gian hoàn thành theo từng xe
    vector<double> route_drone_violation; // vi phạm theo từng xe
    vector<double> route_waiting_violation; // vi phạm chờ theo từng xe

    Solution(): makespan(0), drone_violation(0), waiting_violation(0), fitness(DBL_MAX), is_feasible(true) {}
};

struct TabuMove {
    MoveType type; // 1-0, 1-1, 2-0, 2-1, 2-2, 2-opt
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
    double waiting_violation;
};

vector<vector<double>> distances;
vector<vector<double>> truck_times;
vector<vector<double>> drone_times;
vector<Node> C1; // customers served only by technicians
vector<Node> C2; // customers served by drones or technicians
vector<VehicleFamily> vehicles;
unordered_map<int, int> base_type_by_node;
unordered_map<int, double> base_limit_wait_by_node;
vector<double> base_limit_wait_vec; // O(1) lookup by node id

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

// Adaptive parameters
static constexpr int NUM_MOVE_TYPES = 6; // MT_1_0..MT_2OPT
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

    for (int i = 0; i < NUM_MOVE_TYPES; i++){
        cumulate += weights[i];
        if (r <= cumulate) return i;
    }
    return NUM_MOVE_TYPES - 1; 
}

void update_weights(){
    for (int i = 0; i < NUM_MOVE_TYPES; i++){
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
    base_limit_wait_by_node.clear();
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
        double limit_wait = 60.0;
        static int id = 1;
        ss >> x >> y >> demand;
        nodes.push_back({id++,x,y,demand,limit_wait});
    }
    file.close();

    cout << "Read " << nodes.size() << " nodes (including depot)." << endl;
    if (nodes.size() > 1000) {
        // Bộ rất lớn (> 1000)
        MAX_ITER = 50000;
        //SEGMENT_LENGTH = 5000;
    }
    else if (nodes.size() >= 1000) {
        // Bộ 1000 (501-1000)
        MAX_ITER = 200000;
        //SEGMENT_LENGTH = 2500;
    }
    else if (nodes.size() >= 500) {
        // Bộ 500 (201-500)
        MAX_ITER = 45000;
        //SEGMENT_LENGTH = 1250;
    }
    else if (nodes.size() >= 200) {
        // Bộ 200 (101-200)
        MAX_ITER = 18000;
        //SEGMENT_LENGTH = 600;
    }
    else if (nodes.size() >= 100) {
        // Bộ 100 (100)
        MAX_ITER = 9000;
        //SEGMENT_LENGTH = 100;
    }
    else if (nodes.size() >= 50) {
        // Bộ 50 (50-99)
        MAX_ITER = 4500;
        //SEGMENT_LENGTH = 50;
    }
    else {
        // Bộ nhỏ (6-49)
        MAX_ITER = 4500;
        //SEGMENT_LENGTH = 50;
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

    // Build O(1) limit_wait lookup vector
    base_limit_wait_vec.assign(nodes.size(), 60.0);
    for (const auto& node : nodes) {
        if (node.id >= 0 && node.id < (int)base_limit_wait_vec.size())
            base_limit_wait_vec[node.id] = node.limit_wait;
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
    base_type_by_node.clear();
    for (const auto& node : nodes){
        if (node.id == depot_id) continue;
        base_limit_wait_by_node[node.id] = node.limit_wait;
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
    cout << "Waiting violation: " << sol.waiting_violation << endl;
    cout << "Fitness: " << sol.fitness << endl;
}

void normalize_route(vector<int> &route);
double get_limit_wait_for_node(int node_id);

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
    double drone_violation = 0.0;
    double waiting_violation = 0.0;

    static thread_local vector<pair<int,double>> served_in_trip;
    served_in_trip.clear();
    if (served_in_trip.capacity() < route.size()) served_in_trip.reserve(route.size());

    for (int cid : route) {
        if (cid == depot_id) {
            if (prev != depot_id) current_time += time_matrix[prev][depot_id];
            double arrival_depot = current_time;
            double flight_time = arrival_depot - depart_time;
            if (vehicle.is_drone && flight_time > vehicle.limit_drone) {
                drone_violation += (flight_time - vehicle.limit_drone);
            }

            const double LIMIT_WAIT = 60.0;
            int n_served = (int)served_in_trip.size();
            for (int k = n_served - 1; k >= 0; k--) {
                double wait_time = arrival_depot - served_in_trip[k].second;
                double viol = wait_time - LIMIT_WAIT;
                if (viol > 0.0) {
                    waiting_violation += viol * (k + 1);
                    break;
                }
            }

            depart_time = current_time;
            served_in_trip.clear();
            prev = depot_id;
        } else {
            double travel = time_matrix[prev][cid];
            double entry_time = current_time + travel;
            served_in_trip.push_back({cid, entry_time});
            current_time += travel;
            prev = cid;
        }
    }

    return {current_time, drone_violation, waiting_violation};
}

void evaluate_solution(Solution &sol) {
    for (auto &route : sol.route) normalize_route(route);

    sol.makespan = 0;
    sol.drone_violation = 0;
    sol.waiting_violation = 0;
    sol.fitness = 0;
    sol.is_feasible = true;

    sol.route_time.assign(sol.route.size(), 0.0);
    sol.route_drone_violation.assign(sol.route.size(), 0.0);
    sol.route_waiting_violation.assign(sol.route.size(), 0.0);

    for (size_t i = 0; i < sol.route.size(); i++){
        RouteEval eval = evaluate_route(sol.route[i], vehicles[i]);
        sol.route_time[i] = eval.time;
        sol.route_drone_violation[i] = eval.drone_violation;
        sol.route_waiting_violation[i] = eval.waiting_violation;
        sol.makespan = max(sol.makespan, eval.time);
        sol.drone_violation += eval.drone_violation;
        sol.waiting_violation += eval.waiting_violation;
    }

    if (sol.drone_violation > EPSILON || sol.waiting_violation > EPSILON) sol.is_feasible = false;

    sol.fitness = sol.makespan + alpha1*sol.drone_violation + alpha2*sol.waiting_violation; // include waiting violation
}

void recompute_solution_from_cache(Solution &sol) {
    sol.makespan = 0.0;
    sol.drone_violation = 0.0;
    sol.waiting_violation = 0.0;
    for (size_t i = 0; i < sol.route_time.size(); i++) {
        sol.makespan = max(sol.makespan, sol.route_time[i]);
        sol.drone_violation += sol.route_drone_violation[i];
        if (i < sol.route_waiting_violation.size()) sol.waiting_violation += sol.route_waiting_violation[i];
    }
    sol.is_feasible = (sol.drone_violation <= EPSILON && sol.waiting_violation <= EPSILON);
    sol.fitness = sol.makespan + alpha1 * sol.drone_violation + alpha2 * sol.waiting_violation;
}

void recompute_solution_for_routes(Solution &sol, size_t v1, size_t v2, bool has_second) {
    if (sol.route_time.size() != sol.route.size() || sol.route_drone_violation.size() != sol.route.size() || sol.route_waiting_violation.size() != sol.route.size()) {
        evaluate_solution(sol);
        return;
    }
    RouteEval eval1 = evaluate_route(sol.route[v1], vehicles[v1]);
    sol.route_time[v1] = eval1.time;
    sol.route_drone_violation[v1] = eval1.drone_violation;
    sol.route_waiting_violation[v1] = eval1.waiting_violation;
    if (has_second && v2 != v1) {
        RouteEval eval2 = evaluate_route(sol.route[v2], vehicles[v2]);
        sol.route_time[v2] = eval2.time;
        sol.route_drone_violation[v2] = eval2.drone_violation;
        sol.route_waiting_violation[v2] = eval2.waiting_violation;
    }
    recompute_solution_from_cache(sol);
}

void recompute_solution_for_route(Solution &sol, size_t v1) {
    recompute_solution_for_routes(sol, v1, v1, false);
}

Solution init_greedy_solution() {
    Solution sol;
    sol.route.resize(vehicles.size());

    // ---- Pool khách hàng chưa được phục vụ ----
    vector<int> unvisited_C1, unvisited_C2; // C1: chỉ truck; C2: truck hoặc drone
    for (const auto& n : C1) unvisited_C1.push_back(n.id);
    for (const auto& n : C2) unvisited_C2.push_back(n.id);

    struct Candidate { int cid = -1; int pool = -1; int idx = -1; bool found = false; };

    // Tìm ứng viên xa nhất (want_farthest=true) hoặc gần nhất (false) hợp lệ cho 1 xe
    auto find_candidate = [&](int current_pos, const VehicleFamily& vehicle, bool want_farthest) -> Candidate {
        const auto& M = vehicle.is_drone ? drone_times : truck_times;
        double best_val = want_farthest ? -1.0 : DBL_MAX;
        Candidate res;

        if (!vehicle.is_drone) { // drone không được nhận C1
            for (size_t i = 0; i < unvisited_C1.size(); i++) {
                int cid = unvisited_C1[i];
                double d = M[current_pos][cid];
                if ((want_farthest && d > best_val) || (!want_farthest && d < best_val)) {
                    best_val = d;
                    res = {cid, 1, (int)i, true};
                }
            }
        }
        for (size_t i = 0; i < unvisited_C2.size(); i++) {
            int cid = unvisited_C2[i];
            double d = M[current_pos][cid];
            if ((want_farthest && d > best_val) || (!want_farthest && d < best_val)) {
                best_val = d;
                res = {cid, 2, (int)i, true};
            }
        }
        return res;
    };

    auto remove_from_pool = [&](const Candidate& c) {
        if (c.pool == 1) unvisited_C1.erase(unvisited_C1.begin() + c.idx);
        else if (c.pool == 2) unvisited_C2.erase(unvisited_C2.begin() + c.idx);
    };

    const size_t n_vehicles = vehicles.size();
    vector<int> current_pos(n_vehicles, depot_id);
    vector<double> current_time(n_vehicles, 0.0);  // đồng hồ chạy xuyên suốt cả route (nhiều trip nối tiếp)
    vector<double> depart_time(n_vehicles, 0.0);   // thời điểm rời depot của trip hiện tại
    vector<vector<pair<int,double>>> served_in_trip(n_vehicles); // (id, entry_time) của trip hiện tại
    vector<bool> stopped(n_vehicles, false);

    for (size_t v = 0; v < n_vehicles; v++) sol.route[v].push_back(depot_id);

    vector<size_t> phase1_order;
    for (size_t v = 0; v < n_vehicles; v++) if (vehicles[v].is_drone) phase1_order.push_back(v);
    for (size_t v = 0; v < n_vehicles; v++) if (!vehicles[v].is_drone) phase1_order.push_back(v);

    for (size_t v : phase1_order) {
        const VehicleFamily& vehicle = vehicles[v];
        const auto& M = vehicle.is_drone ? drone_times : truck_times;

        Candidate cand = find_candidate(depot_id, vehicle, true); // điểm xa nhất từ depot
        if (!cand.found) continue; // hết pool phù hợp cho loại xe này -> đành chịu (không thể tránh được)

        double travel_time = M[depot_id][cand.cid];
        sol.route[v].push_back(cand.cid);
        served_in_trip[v].push_back({cand.cid, travel_time});
        current_time[v] = travel_time;
        current_pos[v] = cand.cid;
        remove_from_pool(cand);
    }

    for (size_t v = 0; v < n_vehicles; v++) {
        const VehicleFamily& vehicle = vehicles[v];
        const auto& M = vehicle.is_drone ? drone_times : truck_times;
        vector<int>& route = sol.route[v];

        while (!stopped[v]) {
            bool trip_just_started = served_in_trip[v].empty();
            Candidate cand = find_candidate(current_pos[v], vehicle, trip_just_started);

            if (!cand.found) break; // hết khách hợp lệ cho xe này -> dừng hẳn

            double travel_time = M[current_pos[v]][cand.cid];
            double tentative_time = current_time[v] + travel_time;
            double entry_time = tentative_time;

            // Giả lập: nếu về thẳng depot ngay sau khi ghé điểm này
            double arrival_depot_if_return = tentative_time + M[cand.cid][depot_id];
            double flight_time_if_return = arrival_depot_if_return - depart_time[v];

            bool violate_drone_limit = vehicle.is_drone &&
                ((flight_time_if_return - vehicle.limit_drone) > EPSILON);

            bool violate_wait = false;
            for (auto& se : served_in_trip[v]) {
                double wait_time = arrival_depot_if_return - se.second;
                double limit = get_limit_wait_for_node(se.first);
                if ((wait_time - limit) > EPSILON) { violate_wait = true; break; }
            }

            bool must_force_accept = trip_just_started; // luôn phục vụ được >=1 điểm/trip, tránh kẹt vô hạn

            if ((violate_drone_limit || violate_wait) && !must_force_accept) {
                // Đóng trip hiện tại (KHÔNG thêm candidate)
                double arrival_depot_close = current_time[v] + M[current_pos[v]][depot_id];
                route.push_back(depot_id);
                current_time[v] = arrival_depot_close;
                depart_time[v] = current_time[v];
                served_in_trip[v].clear();
                current_pos[v] = depot_id;

                if (!vehicle.is_drone) {
                    stopped[v] = true; // truck: dừng hẳn công việc tại depot
                }
                continue; // candidate được giữ lại trong pool cho trip/xe kế tiếp
            }

            // Chấp nhận candidate (không vi phạm, hoặc bắt buộc chấp nhận vì là điểm đầu trip)
            route.push_back(cand.cid);
            served_in_trip[v].push_back({cand.cid, entry_time});
            current_time[v] = tentative_time;
            current_pos[v] = cand.cid;
            remove_from_pool(cand);
        }

        // Đóng trip cuối cùng nếu chưa về depot
        if (route.back() != depot_id) {
            current_time[v] += M[current_pos[v]][depot_id];
            route.push_back(depot_id);
        }
    }

    auto try_insert_leftover = [&](int cid, bool is_c1) {
        double best_cost = DBL_MAX;
        int best_v = -1;

        for (size_t v = 0; v < vehicles.size(); v++) {
            if (is_c1 && vehicles[v].is_drone) continue; // drone không được nhận C1
            const auto& M = vehicles[v].is_drone ? drone_times : truck_times;
            vector<int>& route = sol.route[v];
            if (route.size() < 2) continue; // cần ít nhất [depot, depot]

            int prev = route[route.size() - 2]; // điểm ngay trước depot cuối
            double cost = M[prev][cid] + M[cid][depot_id] - M[prev][depot_id];
            if (cost < best_cost) {
                best_cost = cost;
                best_v = (int)v;
            }
        }

        if (best_v != -1) {
            vector<int>& route = sol.route[best_v];
            route.insert(route.end() - 1, cid); // chèn ngay trước depot cuối cùng
        }
    };

    for (int cid : unvisited_C1) try_insert_leftover(cid, true);
    for (int cid : unvisited_C2) try_insert_leftover(cid, false);

    for (size_t v = 0; v < vehicles.size(); v++) normalize_route(sol.route[v]);

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
    auto it = base_type_by_node.find(nid);
    if (it != base_type_by_node.end()) return it->second;
    return -1;
}

double get_limit_wait_for_node(int node_id) {
    if (node_id >= 0 && node_id < (int)base_limit_wait_vec.size())
        return base_limit_wait_vec[node_id];
    return 60.0;
}

bool is_tabu(const vector<TabuMove> &tabu_list, const TabuMove &move){
    for (const auto &tabu_move : tabu_list){
        if (tabu_move.type == move.type && tabu_move.tenure > 0){
            if (move.type == MT_1_1){
                if (((tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id3 == move.customer_id3) ||
                     (tabu_move.customer_id1 == move.customer_id3 && tabu_move.customer_id3 == move.customer_id1)) &&
                    ((tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2) ||
                     (tabu_move.vehicle1 == move.vehicle2 && tabu_move.vehicle2 == move.vehicle1))) {
                    return true;
                }
            }
            else if (move.type == MT_1_0){
                if (tabu_move.customer_id1 == move.customer_id1 &&
                    tabu_move.vehicle1 == move.vehicle1 &&
                    tabu_move.vehicle2 == move.vehicle2) {
                    return true;
                }
            } else if (move.type == MT_2_0){
                if (tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id2 == move.customer_id2
                    && tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2 ) {
                        return true;
                }
            } else if (move.type == MT_2_1){
                if (((tabu_move.customer_id1 == move.customer_id1 && tabu_move.customer_id2 == move.customer_id2 && tabu_move.customer_id3 == move.customer_id3) ||
                        (tabu_move.customer_id1 == move.customer_id3 && tabu_move.customer_id3 == move.customer_id1 && tabu_move.customer_id4 == move.customer_id2)) &&
                        ((tabu_move.vehicle1 == move.vehicle1 && tabu_move.vehicle2 == move.vehicle2) ||
                        (tabu_move.vehicle1 == move.vehicle2 && tabu_move.vehicle2 == move.vehicle1))) {
                        return true;
                    }
            } else if (move.type == MT_2_2){
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
            } else if (move.type == MT_2OPT){
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

Solution move_2_1(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, int cnt_v1=-1, int cnt_v2=-1){
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
    int customer_count_v1 = (cnt_v1 >= 0) ? cnt_v1 : (int)new_sol.route[v1].size() - 2;
    if (customer_count_v1 <= 2) return current_sol;
    int customer_count_v2 = (cnt_v2 >= 0) ? cnt_v2 : (int)new_sol.route[v2].size() - 2;
    if (customer_count_v2 <= 1) return current_sol;
    
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

Solution move_2_2(Solution current_sol, size_t v1, size_t pos1, size_t v2, size_t pos2, int cnt_v1=-1, int cnt_v2=-1){
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

    int customer_count_v1 = (cnt_v1 >= 0) ? cnt_v1 : (int)new_sol.route[v1].size() - 2;
    int customer_count_v2 = (cnt_v2 >= 0) ? cnt_v2 : (int)new_sol.route[v2].size() - 2;
    if (customer_count_v1 <= 2 || customer_count_v2 <= 2) return current_sol;
    
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
    Solution initial_sol = init_greedy_solution();
    Solution best_sol = initial_sol;
    Solution current_sol = initial_sol;

    vector<TabuMove> tabu_list; // danh sách các move bị tabu

    // move_types removed: using MoveType enum
    
    for (int iter = 0; iter < MAX_ITER; iter++){
        double best_Neighbor_fitness = DBL_MAX;
        Solution best_Neighbor_sol = current_sol;
        double current_fitness = current_sol.fitness;
        TabuMove best_move;
        int best_move_node1 = -1, best_move_node2 = -1, best_move_node3 = -1, best_move_node4 = -1;
        bool improved = false;
        //string move_type = "2-2";

        int move_type_idx = select_move_type();
        //int move_type_idx = rand() % MOVE_SET.size();
        MoveType move_type = static_cast<MoveType>(move_type_idx);
        used_count[move_type_idx]++;
        // route đã normalize: size-2 = số customers
        vector<int> customer_count_per_vehicle(current_sol.route.size(), 0);
        for (size_t v = 0; v < current_sol.route.size(); v++) {
            int sz = (int)current_sol.route[v].size();
            customer_count_per_vehicle[v] = (sz >= 2) ? sz - 2 : 0;
        }

        // move 1-0
        if (move_type == MT_1_0) {
            for (size_t v1 = 0; v1 < current_sol.route.size(); v1++) {
                for (size_t pos1 = 1; pos1 < current_sol.route[v1].size()-1; pos1++) {
                    int n1 = current_sol.route[v1][pos1];
                    if (n1 == depot_id) continue;
                    int customer_count_v1 = customer_count_per_vehicle[v1];
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
                            TabuMove move = {MT_1_0, n1, -1, -1, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
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
        if (move_type == MT_1_1) {
            for (size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for (size_t pos1 = 1; pos1 < current_sol.route[v1].size() -1 ; pos1++) {
                    int n1 = current_sol.route[v1][pos1];
                    if (n1 == depot_id) continue;
                    for (size_t v2 = 0; v2 < vehicles.size(); v2++) {
                        for (size_t pos2 = 1; pos2 < current_sol.route[v2].size()-1; pos2++) {
                            int n2 = current_sol.route[v2][pos2];
                            if (n2 == depot_id || n1 == n2 || get_type(n1) != get_type(n2) || ((abs(int(pos1)-int(pos2)) <= 1) && (v1 == v2))) continue;

                            Solution new_sol = move_1_1(current_sol, v1, pos1, v2, pos2);
                            TabuMove move = {MT_1_1, n1, -1, n2, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
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

        if (move_type == MT_2_0) {
            for(size_t v1 = 0; v1 < vehicles.size(); v1++){
                for(size_t pos1 = 1; pos1 < current_sol.route[v1].size()-2; pos1++){
                    int n1 = current_sol.route[v1][pos1];
                    int n2 = current_sol.route[v1][pos1+1];
                    if (n1 == depot_id || n2 == depot_id) continue;
                    if (customer_count_per_vehicle[v1] <= 2) continue;
                    for (size_t v2 = 0; v2 < vehicles.size(); v2++){
                        if (v1 == v2) continue;
                        if ((get_type(n1) == 1 || get_type(n2) == 1) && vehicles[v2].is_drone) continue;
                        for (size_t pos2 = 1; pos2 <= current_sol.route[v2].size(); pos2++){
                            if (pos2 == current_sol.route[v2].size() && !vehicles[v2].is_drone) {
                                continue;
                            }

                            Solution new_sol = move_2_0(current_sol, v1, pos1, v2, pos2);
                            TabuMove move = {MT_2_0, n1, n2, -1, -1, (int)v1, (int)v2, (int)pos1, (int)pos1+1, (int)pos2, (int)pos2+1, TABU_TENURE};
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
        if (move_type == MT_2_1) {
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
                            Solution new_sol = move_2_1(current_sol, v1, pos1, v2, pos2, customer_count_per_vehicle[v1], customer_count_per_vehicle[v2]);
                            TabuMove move = {MT_2_1, n1, n2, n3, -1, (int)v1, (int)v2, (int)pos1, (int)pos1+1, (int)pos2, -1, TABU_TENURE};
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

        if (move_type == MT_2_2){
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

                            Solution new_sol = move_2_2(current_sol, v1, pos1, v2, pos2, customer_count_per_vehicle[v1], customer_count_per_vehicle[v2]);
                            TabuMove move = {MT_2_2, n1, n2, n3, n4, int(v1), int(v2), int(pos1), int(pos1+1), int(pos2), int(pos2+1), TABU_TENURE};
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
        if (move_type == MT_2OPT) {
            // Intra-route 2-opt (cùng xe)
            for(size_t v1 = 0; v1 < vehicles.size(); v1++) {
                for(size_t pos1 = 1; pos1 < current_sol.route[v1].size() - 1; pos1++) {
                    if (current_sol.route[v1][pos1] == depot_id) continue;
                    for(size_t pos2 = pos1 + 2; pos2 < current_sol.route[v1].size() - 1; pos2++) {
                        if (current_sol.route[v1][pos2] == depot_id) continue;

                        int customer_at_pos1 = current_sol.route[v1][pos1];
                        int customer_at_pos2 = current_sol.route[v1][pos2];

                        Solution new_sol = move_2opt(current_sol, v1, pos1, v1, pos2); // Cùng xe v1
                        TabuMove move = {MT_2OPT, customer_at_pos1, -1, customer_at_pos2, -1, (int)v1, (int)v1, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
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
                            TabuMove move = {MT_2OPT, customer_at_pos1, -1, customer_at_pos2, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
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

        if (move_type == MT_1_0) {
            should_apply_move = (best_move_node1 != -1);
        }
        else if (move_type == MT_1_1) {
            should_apply_move = (best_move_node1 != -1 && best_move_node3 != -1);
        }
        else if (move_type == MT_2_0) {
            should_apply_move = (best_move_node1 != -1 && best_move_node2 != -1);
        }
        else if (move_type == MT_2_1) {
            should_apply_move = (best_move_node1 != -1 && best_move_node2 != -1 && best_move_node3 != -1);
        }
        else if (move_type == MT_2_2) {
            should_apply_move = (best_move_node1 != -1 && best_move_node2 != -1 && best_move_node3 != -1 && best_move_node4 != -1);
        }
        else if (move_type == MT_2OPT) {
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
            for (auto& tm : tabu_list) tm.tenure--;
            tabu_list.erase(
                std::remove_if(tabu_list.begin(), tabu_list.end(),
                    [](const TabuMove& m){ return m.tenure <= 0; }),
                tabu_list.end());
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
            if (move_type == MT_1_0) {
                reverse_move = {MT_1_0, best_move_node1, -1, -1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, -1, TABU_TENURE};
            }
            else if (move_type == MT_1_1) {
                reverse_move = {MT_1_1, best_move_node3, -1, best_move_node1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, -1, TABU_TENURE};
            }
            else if (move_type == MT_2_0) {
                reverse_move = {MT_2_0, best_move_node1, best_move_node2, -1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, best_move.pos4, best_move.pos1, best_move.pos2, TABU_TENURE};
            }
            else if (move_type == MT_2_1) {
                reverse_move = {MT_2_1, best_move_node3, -1, best_move_node1, best_move_node2, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, best_move.pos2, TABU_TENURE};
            }
            else if (move_type == MT_2_2) {
                reverse_move = {MT_2_2, best_move_node3, best_move_node4, best_move_node1, best_move_node2, best_move.vehicle2, best_move.vehicle1, best_move.pos3, best_move.pos4, best_move.pos1, best_move.pos2, TABU_TENURE};
            }
            else if (move_type == MT_2OPT) {
                reverse_move = {MT_2OPT, best_move_node3, -1, best_move_node1, -1, best_move.vehicle2, best_move.vehicle1, best_move.pos3, -1, best_move.pos1, -1, TABU_TENURE};
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
    srand(time(nullptr));
    string dataset_path;

    if (argc > 1) {
        dataset_path = argv[1];
    } else {
        dataset_path = "D:\\New folder\\instances\\100.40.2.txt"; 
    }

    read_dataset(dataset_path);

    if (argc > 2) {
        int override_max_iter = atoi(argv[2]);
        if (override_max_iter > 0) {
            MAX_ITER = override_max_iter;
        }
    }

    cout << "\n=== CONFIGURATION ===" << endl;
    cout << "MAX_ITER: " << MAX_ITER << endl;

    printf(" %d\n", MAX_ITER);
 
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

    return 0;
}