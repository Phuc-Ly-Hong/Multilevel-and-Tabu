#include<bits/stdc++.h>
using namespace std;

// NEO/TSPLIB input and comparison adapter. Search below is copied from
// Multilevel_Tabu_no_lwt.cpp, with CVRP objective/feasibility substitutions.
#include <filesystem>
namespace neo {
namespace fs = std::filesystem;
using Routes = std::vector<std::vector<int>>;
struct Instance {
    std::string name, metric, format;
    int n=0, depot=-1, k=0;
    double capacity=0;
    std::vector<double> demand;
    std::vector<std::pair<double,double>> xy;
    std::vector<std::vector<double>> distance;
    fs::path path;
};
static void check(bool ok,const std::string &message) {if(!ok) throw std::runtime_error(message);}
static std::string trim(std::string s) {
    auto a=s.find_first_not_of(" \t\r\n");
    return a==std::string::npos ? "" : s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
static std::string upper(std::string s) {for(char &c:s)c=char(std::toupper(static_cast<unsigned char>(c)));return s;}
static Instance read(const fs::path &path) {
    std::ifstream f(path);check(bool(f),"Cannot open instance: "+path.string());
    Instance p;p.path=path;p.name=path.stem().string();
    std::map<std::string,std::string> h;
    std::map<std::string,std::vector<std::string>> sections;
    std::string section,line;
    while(std::getline(f,line)) {
        line=trim(line);if(line.empty())continue;if(upper(line)=="EOF")break;
        if(line.size()>=8 && line.substr(line.size()-8)=="_SECTION") {section=line;continue;}
        if(!section.empty()) {sections[section].push_back(line);continue;}
        auto pos=line.find(':');check(pos!=std::string::npos,"Invalid TSPLIB header");
        h[upper(trim(line.substr(0,pos)))]=trim(line.substr(pos+1));
    }
    check(h["TYPE"]=="CVRP","Expected TYPE CVRP");
    p.n=std::stoi(h.at("DIMENSION"));p.capacity=std::stod(h.at("CAPACITY"));
    p.metric=upper(h.at("EDGE_WEIGHT_TYPE"));p.format=upper(h["EDGE_WEIGHT_FORMAT"]);
    check(p.n>=2 && p.capacity>0,"Invalid dimension/capacity");
    p.demand.assign(p.n,-1);p.xy.assign(p.n,{NAN,NAN});
    for(const auto &row:sections["DEMAND_SECTION"]) {
        std::istringstream in(row);int id;double q;
        check(bool(in>>id>>q) && id>=1 && id<=p.n && q>=0,"Invalid demand row");
        check(p.demand[id-1]<0,"Duplicate demand node");p.demand[id-1]=q;
    }
    for(const auto &row:sections["NODE_COORD_SECTION"]) {
        std::istringstream in(row);int id;double x,y;
        check(bool(in>>id>>x>>y) && id>=1 && id<=p.n && std::isfinite(x) && std::isfinite(y),"Invalid coordinate row");
        check(!std::isfinite(p.xy[id-1].first),"Duplicate coordinate node");p.xy[id-1]={x,y};
    }
    for(const auto &row:sections["DEPOT_SECTION"]) {
        std::istringstream in(row);int id;
        while(in>>id) {if(id==-1)break;check(p.depot==-1 && id>=1 && id<=p.n,"Expected one depot");p.depot=id-1;}
    }
    check(p.depot==0 && p.demand[0]==0,"NEO numbering requires depot VRP ID 1");
    for(double q:p.demand)check(q>=0 && q<=p.capacity,"Missing or oversized demand");
    std::smatch match;std::string name=path.stem().string();
    if(std::regex_search(name,match,std::regex("-k([0-9]+)",std::regex::icase)))p.k=std::stoi(match[1]);
    else if(h.count("VEHICLES"))p.k=std::stoi(h["VEHICLES"]);
    else if(std::regex_search(h["COMMENT"],match,std::regex("(?:No of trucks|Min no of trucks)\\s*:\\s*([0-9]+)",std::regex::icase)))p.k=std::stoi(match[1]);
    check(p.k>0 && p.k<p.n,"Cannot infer number of vehicles");
    check(std::accumulate(p.demand.begin(),p.demand.end(),0.0)<=p.k*p.capacity,"Insufficient fleet capacity");
    p.distance.assign(p.n,std::vector<double>(p.n));
    if(p.metric=="EXPLICIT") {
        std::vector<double> values;for(const auto &row:sections["EDGE_WEIGHT_SECTION"]) {
            std::istringstream in(row);double v;while(in>>v){check(v>=0 && std::isfinite(v),"Invalid edge weight");values.push_back(v);}
        }
        size_t at=0;auto next=[&](){check(at<values.size(),"Truncated edge weights");return values[at++];};
        if(p.format=="FULL_MATRIX")for(int i=0;i<p.n;++i)for(int j=0;j<p.n;++j)p.distance[i][j]=next();
        else if(p.format=="LOWER_DIAG_ROW" || p.format=="LOWER_ROW") {
            for(int i=0;i<p.n;++i)for(int j=0;j<i+(p.format=="LOWER_DIAG_ROW");++j)p.distance[i][j]=p.distance[j][i]=next();
        } else if(p.format=="UPPER_ROW") {
            for(int i=0;i<p.n;++i)for(int j=i+1;j<p.n;++j)p.distance[i][j]=p.distance[j][i]=next();
        } else throw std::runtime_error("Unsupported EDGE_WEIGHT_FORMAT: "+p.format);
        check(at==values.size(),"Extra edge weights");
    } else {
        check(p.metric=="EUC_2D" || p.metric=="GEO","Unsupported EDGE_WEIGHT_TYPE: "+p.metric);
        for(auto xy:p.xy)check(std::isfinite(xy.first) && std::isfinite(xy.second),"Missing coordinates");
        auto rad=[](double v){int deg=int(v);return std::acos(-1.0)*(deg+5*(v-deg)/3)/180;};
        for(int i=0;i<p.n;++i)for(int j=i+1;j<p.n;++j) {
            auto [x,y]=p.xy[i];auto [a,b]=p.xy[j];double cost;
            if(p.metric=="GEO") {
                double q1=std::cos(rad(y)-rad(b)),q2=std::cos(rad(x)-rad(a)),q3=std::cos(rad(x)+rad(a));
                cost=std::floor(6378.388*std::acos(std::clamp(((1+q1)*q2-(1-q1)*q3)/2,-1.0,1.0))+1);
            } else cost=std::floor(std::hypot(x-a,y-b)+0.5);
            p.distance[i][j]=p.distance[j][i]=cost;
        }
    }
    for(int i=0;i<p.n;++i)for(int j=0;j<p.n;++j)
        check((i!=j || p.distance[i][j]==0) && p.distance[i][j]==p.distance[j][i],"Expected symmetric CVRP distances");
    return p;
}
struct Validation {bool feasible=false;double cost=0;std::vector<double> loads;std::string reason;};
static Validation validate(const Instance &p,const Routes &routes) {
    Validation v;std::vector<int> seen(p.n);int used=0;
    for(const auto &r:routes) {
        if(r.empty())continue;
        ++used;
        int prev=p.depot;double load=0;
        for(int node:r) {
            if(node<0 || node>=p.n || node==p.depot){v.reason="Invalid customer";return v;}
            ++seen[node];load+=p.demand[node];v.cost+=p.distance[prev][node];prev=node;
        }
        v.cost+=p.distance[prev][p.depot];v.loads.push_back(load);if(load>p.capacity)v.reason="Capacity exceeded";
    }
    if(used>p.k)v.reason="Fleet exceeded";
    for(int i=0;i<p.n;++i)if(i!=p.depot && seen[i]!=1)v.reason="Customer coverage/uniqueness failed";
    v.feasible=v.reason.empty();return v;
}
struct Reference {double published=NAN,recomputed=0;std::string status="MISSING";bool valid=false;};
static Reference reference(const Instance &p) {
    auto path=p.path;path.replace_extension(".sol");std::ifstream f(path);Reference ref;if(!f)return ref;
    Routes routes;std::string line;int sections=0;
    while(std::getline(f,line)) {
        line=trim(line);auto u=upper(line);if(u.rfind("SOLUTION",0)==0)++sections;
        if(u.rfind("ROUTE",0)==0 || u.rfind("OUTE",0)==0) {
            auto pos=line.find(':');check(pos!=std::string::npos,"Bad reference route");std::istringstream in(line.substr(pos+1));
            std::vector<int> r;int node;while(in>>node)r.push_back(node);routes.push_back(r);
        } else if(u.rfind("COST",0)==0){auto value=line.substr(4);std::replace(value.begin(),value.end(),':',' ');std::istringstream(value)>>ref.published;}
    }
    auto v=validate(p,routes);ref.valid=v.feasible && sections<=1 && !routes.empty();ref.recomputed=v.cost;
    ref.status=!ref.valid?"REVIEW":std::isfinite(ref.published) && std::abs(ref.published-v.cost)<1e-6?"PASS":"COST_MISMATCH";
    return ref;
}
static std::string quote(const std::string &s){std::string out="\"";for(char c:s){if(c=='\"')out+='\"';out+=c;}return out+'\"';}
// CSV solution uses original VRP node IDs, including depot at both ends.
static std::string solution_json(const Routes &routes) {
    std::ostringstream out;
    out << '[';
    for (size_t r=0; r<routes.size(); ++r) {
        if (r) out << ',';
        out << "[1";
        for (int node:routes[r]) out << ',' << node+1;
        out << ",1]";
    }
    out << ']';
    return out.str();
}
}


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
    double makespan; // CVRP total distance (legacy name); thời gian hoàn thành
    double drone_violation; // CVRP capacity excess (legacy name); tổng số thời gian vi phạm thời gian bay của drone
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

struct MoveEvalResult {
    bool feasible;
    double fitness;
    double makespan;
    double drone_violation;
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

neo::Instance neo_instance;
bool NEO_VERIFY = false;
int NEO_LEVELS_USED = 1;
// A run-wide iteration ceiling shared by ML and its no-multilevel baseline.
long long NEO_ITERATION_BUDGET = -1;
long long NEO_ITERATIONS_USED = 0;
int NEO_TABU_CALLS = 0;

int depot_id = 0;
int num_nodes = 0;
double alpha1 = 1.0; // tham số hàm phạt thứ nhất
double alpha2 = 1.0; // tham số hàm phạt thứ hai
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

// Các tham số dưới đây có thể tune bằng irace (xem tuning/README.md).
double delta1 = 0.3;
double delta2 = 0.2;
double delta3 = 0.1;
double delta4 = 0.3;

double TABU_TENURE_FACTOR = 0.25; // Clamp tenure using customer-scaled bounds.
double tabu1 = 0.10; // Minimum tenure factor per customer.
double tabu2 = 0.25; // Maximum tenure factor per customer.

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
    neo_instance = neo::read(filename);
    C1.clear(); C2.clear(); vehicles.clear(); base_type_by_node.clear();
    merged_nodes_info.clear();
    vector<Node> nodes;
    for (int id = 0; id < neo_instance.n; ++id) {
        auto xy = neo_instance.xy[id];
        Node node = {id, isfinite(xy.first) ? xy.first : 0.0,
                         isfinite(xy.second) ? xy.second : 0.0, id == 0 ? -1.0 : 0.0};
        nodes.push_back(node);
        if (id != 0) { C1.push_back(node); base_type_by_node[id] = 1; }
    }
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
        MAX_ITER = 5000;
        MAX_NO_IMPROVE = 500000;
    }
    else if (nodes.size() >= 200) {
        // Bộ 200 (101-200)
        MAX_ITER = 2000;
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


    // CVRP uses the declared edge weights directly, with no speed conversion.
    base_distance_matrix = neo_instance.distance;
    truck_times = base_distance_matrix;
    drone_times = base_distance_matrix; // compatibility matrix; no drones exist
    num_nodes = neo_instance.n;
    for (int v = 0; v < neo_instance.k; ++v)
        vehicles.push_back({v + 1, 1.0, false, neo_instance.capacity});
}

void print_solution(const Solution &sol){
    cout << "Route details:" << endl;
    for (size_t v = 0; v < sol.route.size(); v++) {
        cout << "Vehicle " << v << ": ";
        for (int cid : sol.route[v]) cout << cid << " ";
        cout << endl;
    }
    cout << "Total distance: " << sol.makespan << endl;
    cout << "Capacity violation: " << sol.drone_violation << endl;
    cout << "Fitness: " << sol.fitness << endl;
}

unordered_map<int, int> node_id_to_index_cache;

void update_node_index_cache(const LevelInfo& level) {
    node_id_to_index_cache.clear();
    node_id_to_index_cache.reserve(level.nodes.size() * 2);
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
    if (tmp.size() == 1) tmp.push_back(depot_id);
    route.swap(tmp);
}

map<int, double> internal_distance_cache;

// Keep RouteEval's legacy names: time = distance, drone_violation = excess load.
RouteEval evaluate_route(vector<int> &route, const VehicleFamily &vehicle, const LevelInfo *current_level = nullptr) {
    normalize_route(route);
    const auto &matrix = current_level ? current_level->truck_time_matrix : truck_times;
    double distance = 0.0, load = 0.0;
    int prev = depot_id;
    for (size_t pos=0; pos<route.size(); ++pos) {
        int cid=route[pos];
        // CVRP is one depot-to-depot trip per vehicle; interior depots are invalid.
        if (cid==depot_id && pos>0 && pos+1<route.size()) return {DBL_MAX/100, DBL_MAX/100};
        if (current_level) {
            int from=find_node_index_fast(prev), to=find_node_index_fast(cid);
            neo::check(from>=0 && to>=0,"Unknown node at active level");
            distance+=matrix[from][to];
            if (cid!=depot_id) {
                auto info=merged_nodes_info.find(cid);
                if(info!=merged_nodes_info.end())distance+=info->second.internal_truck_time;
                const auto &originals=current_level->node_mapping.at(cid);
                for(int original:originals)load+=neo_instance.demand.at(original);
            }
        } else {
            distance+=matrix.at(prev).at(cid);
            if(cid!=depot_id)load+=neo_instance.demand.at(cid);
        }
        prev=cid;
    }
    return {distance, max(0.0, load-vehicle.limit_drone)}; // limit_drone stores CVRP capacity
}

void evaluate_solution(Solution &sol, const LevelInfo *current_level = nullptr) {
    for (auto &route : sol.route) normalize_route(route);

    sol.makespan = 0;
    sol.drone_violation = 0;
    sol.fitness = 0;
    sol.is_feasible = true;

    sol.route_time.assign(sol.route.size(), 0.0);
    sol.route_drone_violation.assign(sol.route.size(), 0.0);

    for (size_t i = 0; i < sol.route.size(); i++){
        RouteEval eval = evaluate_route(sol.route[i], vehicles[i], current_level);
        sol.route_time[i] = eval.time;
        sol.route_drone_violation[i] = eval.drone_violation;
        sol.makespan += eval.time; // CVRP total distance
        sol.drone_violation += eval.drone_violation;
    }

    if (sol.drone_violation > EPSILON) sol.is_feasible = false;

    sol.fitness = sol.makespan + alpha1*sol.drone_violation ;
}

void recompute_solution_from_cache(Solution &sol) {
    sol.makespan = 0.0;
    sol.drone_violation = 0.0;
    for (size_t i = 0; i < sol.route_time.size(); i++) {
        sol.makespan += sol.route_time[i]; // CVRP total distance
        sol.drone_violation += sol.route_drone_violation[i];
    }
    sol.is_feasible = (sol.drone_violation <= EPSILON);
    sol.fitness = sol.makespan + alpha1 * sol.drone_violation;
}

void recompute_solution_for_routes(Solution &sol, size_t v1, size_t v2, bool has_second, const LevelInfo *current_level) {
    if (sol.route_time.size() != sol.route.size() || sol.route_drone_violation.size() != sol.route.size()) {
        evaluate_solution(sol, current_level);
        return;
    }
    RouteEval eval1 = evaluate_route(sol.route[v1], vehicles[v1], current_level);
    sol.route_time[v1] = eval1.time;
    sol.route_drone_violation[v1] = eval1.drone_violation;
    if (has_second && v2 != v1) {
        RouteEval eval2 = evaluate_route(sol.route[v2], vehicles[v2], current_level);
        sol.route_time[v2] = eval2.time;
        sol.route_drone_violation[v2] = eval2.drone_violation;
    }
    recompute_solution_from_cache(sol);
}

void recompute_solution_for_route(Solution &sol, size_t v1, const LevelInfo *current_level) {
    recompute_solution_for_routes(sol, v1, v1, false, current_level);
}

MoveEvalResult evaluate_move_routes_delta(const Solution& current_sol,
                                         size_t v1,
                                         size_t v2,
                                         vector<int> route_v1,
                                         vector<int> route_v2,
                                         const LevelInfo* current_level) {
    RouteEval eval1 = evaluate_route(route_v1, vehicles[v1], current_level);
    RouteEval eval2 = (v2 == v1) ? eval1 : evaluate_route(route_v2, vehicles[v2], current_level);

    if (current_sol.route_time.size() != current_sol.route.size() ||
        current_sol.route_drone_violation.size() != current_sol.route.size()) {
        Solution tmp = current_sol;
        tmp.route[v1] = route_v1;
        tmp.route[v2] = route_v2;
        evaluate_solution(tmp, current_level);
        return {tmp.is_feasible, tmp.fitness, tmp.makespan, tmp.drone_violation};
    }

    double new_drone_violation = current_sol.drone_violation
        - current_sol.route_drone_violation[v1]
        + eval1.drone_violation;
    if (v2 != v1) {
        new_drone_violation = new_drone_violation
            - current_sol.route_drone_violation[v2]
            + eval2.drone_violation;
    }

    double new_makespan = 0.0;
    for (size_t i = 0; i < current_sol.route_time.size(); i++) {
        double t = current_sol.route_time[i];
        if (i == v1) t = eval1.time;
        if (i == v2) t = eval2.time;
        new_makespan += t; // CVRP total distance
    }

    bool feasible = (new_drone_violation <= EPSILON);
    double fitness = new_makespan + alpha1 * new_drone_violation;
    return {feasible, fitness, new_makespan, new_drone_violation};
}

int get_type(int nid, const LevelInfo *current_level = nullptr) {
    if (current_level != nullptr) {
        // Dùng level hiện tại
        int idx = find_node_index_fast(nid);
        if (idx != -1) {
            double type_value = current_level->nodes[idx].c1_or_c2;
            if (type_value > 0) return 2;
            if (type_value == 0) return 1;
        }
    } else {
        // Dùng cache từ dataset gốc
        auto it = base_type_by_node.find(nid);
        if (it != base_type_by_node.end()) return it->second;
    }
    return -1;
}


// A feasibility predicate for the greedy choice, not another routing heuristic.
// It only answers whether the remaining demands fit the remaining k capacities.
bool cvrp_capacity_completion(const vector<int> &unserved, int skipped,
                              const vector<double> &loads, int vehicle, int customer) {
    vector<long long> demand, remaining;
    for(size_t i=0;i<unserved.size();++i)if(int(i)!=skipped)demand.push_back(llround(neo_instance.demand[unserved[i]]));
    for(size_t v=0;v<loads.size();++v)remaining.push_back(llround(neo_instance.capacity-loads[v]-(int(v)==vehicle?neo_instance.demand[customer]:0)));
    if(any_of(remaining.begin(),remaining.end(),[](long long q){return q<0;}))return false;
    sort(demand.rbegin(),demand.rend());sort(remaining.rbegin(),remaining.rend());
    unordered_set<string> failed;
    function<bool(size_t,vector<long long>)> fits=[&](size_t i,vector<long long> free) {
        if(i==demand.size())return true;
        if(demand[i]>free.front())return false;
        string key=to_string(i)+":";for(auto q:free)key+=to_string(q)+",";
        if(failed.count(key))return false;
        long long last=-1;
        for(size_t v=0;v<free.size();++v) {
            if(free[v]<demand[i] || free[v]==last)continue;
            last=free[v];auto next=free;next[v]-=demand[i];sort(next.rbegin(),next.rend());
            if(fits(i+1,move(next)))return true;
        }
        failed.insert(move(key));return false;
    };
    return fits(0,remaining);
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

    vector<double> cvrp_load(vehicles.size(), 0.0);
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
                if (cvrp_load[tech_indices[t]] + neo_instance.demand[cid] > neo_instance.capacity + EPSILON) continue;
                if (d < best_time) {
                    if (!cvrp_capacity_completion(unserved_C1, int(i), cvrp_load, tech_indices[t], cid)) continue;
                    best_time = d;
                    best_tech = t;
                    best_cid = cid;
                    best_idx = i;
                }
            }
        }
        neo::check(best_tech >= 0, "No capacity-feasible completion for greedy initialization");
        int vehicle_id = tech_indices[best_tech];
        cvrp_load[vehicle_id] += neo_instance.demand[best_cid];
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

    // Customer allocation logging removed for performance.

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
    (void)end_time;

    return sol;
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
    
    recompute_solution_for_routes(new_sol, v1, v2, true, current_level);
    return new_sol;
}

Solution move_1_1(Solution current_sol, size_t v1, size_t node1, size_t v2, size_t node2, const LevelInfo *current_level){
    Solution new_sol = current_sol;
    int cid1 = new_sol.route[v1][node1];
    int cid2 = new_sol.route[v2][node2];
    if (cid1 == depot_id || cid2 == depot_id) return current_sol; // không di chuyển depot
    swap(new_sol.route[v1][node1], new_sol.route[v2][node2]);
    recompute_solution_for_routes(new_sol, v1, v2, true, current_level);
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

    recompute_solution_for_routes(new_sol, v1, v2, true, current_level);
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

    recompute_solution_for_routes(new_sol, v1, v2, true, current_level);
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

    recompute_solution_for_routes(new_sol, v1, v2, true, current_level);
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
        recompute_solution_for_route(new_sol, v1, current_level);
    } else {
        recompute_solution_for_routes(new_sol, v1, v2, true, current_level);
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


neo::Routes cvrp_original_routes(const Solution &sol,const LevelInfo *level) {
    neo::Routes routes;
    for(const auto &r:sol.route) {
        vector<int> out;
        for(int node:r)if(node!=depot_id) {
            if(level) {const auto &original=level->node_mapping.at(node);out.insert(out.end(),original.begin(),original.end());}
            else out.push_back(node);
        }
        if(!out.empty())routes.push_back(move(out));
    }
    return routes;
}
void cvrp_verify_solution(const Solution &sol,const LevelInfo *level) {
    if(!NEO_VERIFY)return;
    auto validation=neo::validate(neo_instance,cvrp_original_routes(sol,level));
    neo::check(abs(validation.cost-sol.makespan)<EPSILON,"CVRP objective does not match original-node distances");
    if(!validation.feasible)neo::check(validation.reason=="Capacity exceeded","CVRP coverage/fleet invalid");
    neo::check(validation.feasible==sol.is_feasible,"CVRP feasibility mismatch");
    Solution recomputed=sol;evaluate_solution(recomputed,level);
    neo::check(abs(recomputed.fitness-sol.fitness)<EPSILON && abs(recomputed.drone_violation-sol.drone_violation)<EPSILON,"Cached CVRP evaluation mismatch");
}

Solution tabu_search(Solution initial_sol, const LevelInfo *current_level){
    ++NEO_TABU_CALLS;
    const int num_customer = max(1, num_nodes - 1); // Exclude depot; use current level.
    const int min_tenure = max(1, (int)ceil(tabu1 * num_customer));
    const int max_tenure = max(min_tenure, (int)ceil(tabu2 * num_customer));
    TABU_TENURE = max(min_tenure, min((int)ceil(num_customer * TABU_TENURE_FACTOR), max_tenure));
    if (current_level != nullptr) update_node_index_cache(*current_level);

    Solution best_sol = initial_sol;
    Solution current_sol = initial_sol;

    unordered_map<int, int> level_type_cache;
    if (current_level != nullptr) {
        level_type_cache.reserve(current_level->nodes.size() * 2);
        for (const auto& node : current_level->nodes) {
            if (node.id == depot_id) continue;
            level_type_cache[node.id] = (node.c1_or_c2 > 0) ? 2 : 1;
        }
    }
    auto get_type_fast = [&](int nid) -> int {
        if (current_level != nullptr) {
            auto it = level_type_cache.find(nid);
            return (it != level_type_cache.end()) ? it->second : -1;
        }
        auto it = base_type_by_node.find(nid);
        return (it != base_type_by_node.end()) ? it->second : -1;
    };

    vector<TabuMove> tabu_list; // danh sách các move bị tabu
    int no_improve_count = 0;
    int last_depot_opt_iter = 0;

    vector<string> move_types = {"1-0", "1-1", "2-0", "2-1", "2-2", "2-opt"};
    
    for (int iter = 0; iter < MAX_ITER && no_improve_count < MAX_NO_IMPROVE &&
         (NEO_ITERATION_BUDGET < 0 || NEO_ITERATIONS_USED < NEO_ITERATION_BUDGET); iter++){
        ++NEO_ITERATIONS_USED;
        double best_Neighbor_fitness = DBL_MAX;
        Solution best_Neighbor_sol = current_sol;
        bool best_neighbor_ready = false;
        double current_fitness = current_sol.fitness;
        TabuMove best_move;
        int best_move_node1 = -1, best_move_node2 = -1, best_move_node3 = -1, best_move_node4 = -1;
        bool improved = false;
        //string move_type = "2-2";

        int move_type_idx = select_move_type();
        string move_type = MOVE_SET[move_type_idx];
        used_count[move_type_idx]++;

        vector<int> customer_count_per_vehicle(current_sol.route.size(), 0);
        for (size_t v = 0; v < current_sol.route.size(); v++) {
            int count = 0;
            for (int node : current_sol.route[v]) {
                if (node != depot_id) count++;
            }
            customer_count_per_vehicle[v] = count;
        }

        // move 1-0
        if (move_type == "1-0") {
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
                                if (get_type_fast(n1) == 1) continue;
                                if (v1 == v2) continue;
                            } else {
                                if (v1 == v2) continue;
                                if (pos2 == current_sol.route[v2].size() - 1) continue;
                                if (get_type_fast(n1) == 1 && vehicles[v2].is_drone) continue;
                            }

                            vector<int> route_v1 = current_sol.route[v1];
                            vector<int> route_v2 = current_sol.route[v2];
                            int cid = route_v1[pos1];

                            if (pos2 == route_v2.size() && vehicles[v2].is_drone) {
                                route_v1.erase(route_v1.begin() + pos1);
                                route_v2.push_back(cid);
                                if (!route_v2.empty() && route_v2.back() != depot_id) {
                                    route_v2.push_back(depot_id);
                                }
                            } else {
                                route_v1.erase(route_v1.begin() + pos1);
                                route_v2.insert(route_v2.begin() + pos2, cid);
                            }

                            MoveEvalResult eval = evaluate_move_routes_delta(current_sol, v1, v2, route_v1, route_v2, current_level);
                            TabuMove move = {"1-0", n1, -1, -1, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);

                            if (eval.feasible && (eval.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = eval.fitness;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = -1;
                                best_move_node3 = -1;
                                best_move_node4 = -1;
                                best_neighbor_ready = false;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (eval.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = eval.fitness;
                                    best_move = move;
                                    best_move_node1 = n1;   
                                    best_move_node2 = -1;
                                    best_move_node3 = -1;
                                    best_move_node4 = -1;
                                    best_neighbor_ready = false;
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
                            if (n2 == depot_id || n1 == n2 || get_type_fast(n1) != get_type_fast(n2) || ((abs(int(pos1)-int(pos2)) <= 1) && (v1 == v2))) continue;

                            vector<int> route_v1 = current_sol.route[v1];
                            vector<int> route_v2 = current_sol.route[v2];
                            if (v1 == v2) {
                                swap(route_v1[pos1], route_v1[pos2]);
                                route_v2 = route_v1;
                            } else swap(route_v1[pos1], route_v2[pos2]);

                            MoveEvalResult eval = evaluate_move_routes_delta(current_sol, v1, v2, route_v1, route_v2, current_level);
                            TabuMove move = {"1-1", n1, -1, n2, -1, (int)v1, (int)v2, (int)pos1, -1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);

                            if (eval.feasible && (eval.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = eval.fitness;
                                best_move = move;
                                best_move_node1 = n1;
                                best_move_node2 = -1;
                                best_move_node3 = n2;
                                best_move_node4 = -1;
                                best_neighbor_ready = false;
                                improved = true;
                            } else if (improved == false) {
                                if (!tabu && (eval.fitness < best_Neighbor_fitness - EPSILON)) {
                                    best_Neighbor_fitness = eval.fitness;
                                    best_move = move;
                                    best_move_node1 = n1;
                                    best_move_node2 = -1;
                                    best_move_node3 = n2;
                                    best_move_node4 = -1;
                                    best_neighbor_ready = false;
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
                        if ((get_type_fast(n1) == 1 || get_type_fast(n2) == 1) && vehicles[v2].is_drone) continue;
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
                                best_neighbor_ready = true;
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
                                    best_neighbor_ready = true;
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
                            if ((get_type_fast(n1) == 1 || get_type_fast(n2) == 1) && vehicles[v2].is_drone) continue;
                            if (get_type_fast(n3) == 1 && vehicles[v1].is_drone) continue;
                            Solution new_sol = move_2_1(current_sol, v1, pos1, v2, pos2, current_level);
                            TabuMove move = {"2-1", n1, n2, n3, -1, (int)v1, (int)v2, (int)pos1, (int)pos1+1, (int)pos2, -1, TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);
                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_neighbor_ready = true;
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
                                    best_neighbor_ready = true;
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
                            if ((get_type_fast(n1) == 1 || get_type_fast(n2) == 1) && vehicles[v2].is_drone) continue;
                            if ((get_type_fast(n3) == 1 || get_type_fast(n4) == 1) && vehicles[v1].is_drone) continue;

                            Solution new_sol = move_2_2(current_sol, v1, pos1, v2, pos2, current_level);
                            TabuMove move = {"2-2", n1, n2, n3, n4, int(v1), int(v2), int(pos1), int(pos1+1), int(pos2), int(pos2+1), TABU_TENURE};
                            bool tabu = is_tabu(tabu_list, move);
                            if (new_sol.is_feasible && (new_sol.fitness < best_sol.fitness - EPSILON)) {
                                best_Neighbor_fitness = new_sol.fitness;
                                best_Neighbor_sol = new_sol;
                                best_neighbor_ready = true;
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
                                    best_neighbor_ready = true;
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
                            best_neighbor_ready = true;
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
                                best_neighbor_ready = true;
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
                                    if (cid != depot_id && get_type_fast(cid) == 1) {  
                                        invalid_move = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (!invalid_move && vehicles[v1].is_drone) {
                                for (size_t i = pos2; i < current_sol.route[v2].size() - 1; i++) {
                                    int cid = current_sol.route[v2][i];
                                    if (cid != depot_id && get_type_fast(cid) == 1) {  
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
                                best_neighbor_ready = true;
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
                                    best_neighbor_ready = true;
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
            if (best_neighbor_ready) {
                current_sol = best_Neighbor_sol;
            } else if (move_type == "1-0") {
                current_sol = move_1_0(current_sol, best_move.vehicle1, best_move.pos1, best_move.vehicle2, best_move.pos3, current_level);
            } else if (move_type == "1-1") {
                current_sol = move_1_1(current_sol, best_move.vehicle1, best_move.pos1, best_move.vehicle2, best_move.pos3, current_level);
            } else {
                current_sol = best_Neighbor_sol;
            }

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

            cvrp_verify_solution(current_sol, current_level);

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

vector<tuple<double, int, int>> collect_merge_candidates(const LevelInfo& current_level, const Solution& best_solution){
    vector<tuple<double, int, int>> candidates;
    set<pair<int,int>> solution_edges;
    
    for (size_t v = 0; v < best_solution.route.size(); v++) {
        const vector<int>& route = best_solution.route[v];
        for (size_t i = 0; i < route.size() - 1; i++) {
            int from_node = route[i];
            int to_node = route[i + 1];
            
            if (from_node == depot_id || to_node == depot_id) continue;
            
            solution_edges.insert(make_pair(from_node, to_node));
        }
    }

    for (const auto& edge : solution_edges){
        int from_node = edge.first;
        int to_node = edge.second;

        int idx_from = find_node_index_fast(from_node);
        int idx_to = find_node_index_fast(to_node);
        if (idx_from == -1 || idx_to == -1){
            continue;
        }

        const Node& node_from = current_level.nodes[idx_from];
        const Node& node_to = current_level.nodes[idx_to];
        
        bool same_type = (node_from.c1_or_c2 == 0 && node_to.c1_or_c2 == 0) || 
                        (node_from.c1_or_c2 > 0 && node_to.c1_or_c2 > 0);
        
        if (same_type){
            double distance = current_level.truck_time_matrix[idx_from][idx_to];
            candidates.emplace_back(make_tuple(distance, from_node, to_node));
            
        }
    }

    sort(candidates.begin(), candidates.end(), [](const tuple<double, int, int>& a, const tuple<double, int, int>& b) {
        return get<0>(a) < get<0>(b);
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
    
    vector<tuple<double,int,int>> candidates = collect_merge_candidates(current_level, best_solution);
    
    // Số cạnh được chọn để merge theo tỉ lệ nén cấu hình
    int num_to_merge = max(1, (int)(candidates.size() * MERGE_RATIO));
    
    //cout << "\n=== MERGING " << num_to_merge << " / " << candidates.size() << " EDGES (20%) ===" << endl;
    
    unordered_set<int> merged_nodes;
    merged_nodes.reserve(candidates.size() * 2);
    vector<vector<int>> merged_groups;
    merged_groups.reserve(num_to_merge);
    unordered_map<int, int> node_to_group;
    node_to_group.reserve(candidates.size() * 2);
    
    for (int i = 0; i < num_to_merge && i < candidates.size(); i++) {
        double distance = get<0>(candidates[i]);
        int node_a = get<1>(candidates[i]);
        int node_b = get<2>(candidates[i]);

        if (node_a == depot_id || node_b == depot_id) {
            continue;
        }
        
        int group_idx_a = -1;
        int group_idx_b = -1;
        auto it_a = node_to_group.find(node_a);
        if (it_a != node_to_group.end()) group_idx_a = it_a->second;
        auto it_b = node_to_group.find(node_b);
        if (it_b != node_to_group.end()) group_idx_b = it_b->second;

        if (group_idx_a != -1 && group_idx_a == group_idx_b) {
            continue;
        }
        
        // Case 1: Cả 2 đều chưa có trong group nào -> Tạo group mới
        if (group_idx_a == -1 && group_idx_b == -1) {
            merged_groups.push_back({node_a, node_b});
            int new_idx = static_cast<int>(merged_groups.size() - 1);
            node_to_group[node_a] = new_idx;
            node_to_group[node_b] = new_idx;
            merged_nodes.insert(node_a);
            merged_nodes.insert(node_b);
            // Merge logging removed for performance.
        }
        // Case 2: node_a đã có group, node_b chưa -> Thêm node_b vào group của node_a
        else if (group_idx_a != -1 && group_idx_b == -1) {
            auto& group = merged_groups[group_idx_a];
            if (group.empty()) continue;
            // Kiểm tra node a ở đầu hay cuối group
            if (group.back() == node_a){
                group.push_back(node_b);
            } else if (group.front() == node_a){
                group.insert(group.begin(), node_b);
            } else {
                // Không nên xảy ra
                cout << " Warning: node " << node_a << " not at group ends!" << endl;
                continue;
            }
            node_to_group[node_b] = group_idx_a;
            merged_nodes.insert(node_b);
            // Merge logging removed for performance.
        }
        // Case 3: node_b đã có group, node_a chưa -> Thêm node_a vào group của node_b
        else if (group_idx_a == -1 && group_idx_b != -1) {
            auto& group = merged_groups[group_idx_b];
            if (group.empty()) continue;
            if (group.front() == node_b){
                group.insert(group.begin(), node_a);
            } else if (group.back() == node_b){
                group.push_back(node_a);
            } else {
                // Không nên xảy ra
                continue;
            }
            node_to_group[node_a] = group_idx_b;
            merged_nodes.insert(node_a);
        }
        // Case 4: Cả 2 đã có group khác nhau → Merge 2 groups
        else if (group_idx_a != group_idx_b) {
            // Chỉ nối nếu node_a ở cuối group_a VÀ node_b ở đầu group_b
            auto& group_a = merged_groups[group_idx_a];
            auto& group_b = merged_groups[group_idx_b];
            if (group_a.empty() || group_b.empty()) continue;
            if (group_a.back() == node_a && group_b.front() == node_b) {
                // Nối group_b vào cuối group_a
                group_a.insert(
                    group_a.end(),
                    group_b.begin(),
                    group_b.end()
                );
                for (int node : group_b) {
                    node_to_group[node] = group_idx_a;
                }
                group_b.clear();
                // Merge logging removed for performance.
            } else {
                // Merge logging removed for performance.
            }
        }
    }
    
    if (merged_groups.empty()) {
        return current_level;
    }
    
    // Đặt tên mới cho node
    int next_node_id = (next_level.level_id) * 1000;
    
    next_level.nodes.push_back({depot_id, 0.0, 0.0, -1.0});
    next_level.node_mapping[depot_id] = {depot_id};
    
    for (const auto& group : merged_groups) {
        if (group.empty()) continue;
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
                        info.internal_drone_time += drone_edge_time;;
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
            //cout << "Internal distances for merged node " << merged_node.id << ": " << info.internal_distance << endl; 
            
            // ánh xạ node merge về toàn bộ node gốc
            vector<int> original_nodes;
            original_nodes.reserve(group.size());
            unordered_set<int> seen_originals;
            seen_originals.reserve(group.size() * 2);
            for (int node_id : group) {
                auto it = current_level.node_mapping.find(node_id);
                if (it != current_level.node_mapping.end()) {
                    for (int orig : it->second) {
                        if (seen_originals.insert(orig).second) original_nodes.push_back(orig);
                    }
                } else {
                    if (seen_originals.insert(node_id).second) original_nodes.push_back(node_id);
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

    unordered_map<int, int> original_to_new;
    original_to_new.reserve(next_level.node_mapping.size() * 4);
    original_to_new[depot_id] = depot_id;

    for (const auto& entry : next_level.node_mapping) {
        int new_node_id = entry.first;
        if (new_node_id == depot_id) continue;
        for (int orig : entry.second) {
            original_to_new.emplace(orig, new_node_id);
        }
    }
    
    unordered_map<int, int> old_to_new_mapping;
    old_to_new_mapping.reserve(old_level.nodes.size() * 2);
    old_to_new_mapping[depot_id] = depot_id;
    
    for (const auto& old_node : old_level.nodes) {
        int old_node_id = old_node.id;
        if (old_node_id == depot_id) continue;
        int mapped = -1;
        auto old_it = old_level.node_mapping.find(old_node_id);
        if (old_it != old_level.node_mapping.end()) {
            for (int old_orig : old_it->second) {
                auto map_it = original_to_new.find(old_orig);
                if (map_it != original_to_new.end()) {
                    mapped = map_it->second;
                    break;
                }
            }
        } else {
            auto map_it = original_to_new.find(old_node_id);
            if (map_it != original_to_new.end()) {
                mapped = map_it->second;
            }
        }

        if (mapped != -1) {
            old_to_new_mapping[old_node_id] = mapped;
        } else {
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

    unordered_set<int> fine_level_node_ids;
    fine_level_node_ids.reserve(fine_level.nodes.size() * 2);
    for (const auto& node : fine_level.nodes) {
        fine_level_node_ids.insert(node.id);
    }
    
    unordered_map<int, vector<int>> coarse_to_fine;
    coarse_to_fine.reserve(coarse_level.nodes.size() * 2);
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

    // UNMERGE ROUTES
    for (size_t v = 0; v < coarse_sol.route.size(); v++) {
        fine_sol.route[v].push_back(depot_id);
        
        for (int coarse_node_id : coarse_sol.route[v]) {
            if (coarse_node_id == depot_id) fine_sol.route[v].push_back(depot_id);
            
            auto it = coarse_to_fine.find(coarse_node_id);
            if (it != coarse_to_fine.end()) {
                for (int fine_node_id : it->second) {
                    fine_sol.route[v].push_back(fine_node_id);
                }
            }
        }
        
        fine_sol.route[v].push_back(depot_id);
    }
    
    return fine_sol;
}

Solution multilevel_tabu_search() {
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

    while (coarsening) {
        //cout << "\n--- LEVEL " << L << " ---" << endl;
        auto level_start = chrono::high_resolution_clock::now();   
        update_node_index_cache(all_levels[L]);
        Solution s_current = tabu_search(s, &all_levels[L]);
        auto level_end = chrono::high_resolution_clock::now();
        double level_time = chrono::duration<double>(level_end - level_start).count();
        /*if (L >= 3 && s_current.fitness == prev_fitness) {
            break;
        }*/
        prev_fitness = s_current.fitness;
        s = s_current;
        cvrp_verify_solution(s, &all_levels[L]);

        if (L >= max_levels || (NEO_ITERATION_BUDGET >= 0 && NEO_ITERATIONS_USED >= NEO_ITERATION_BUDGET)) {
            break;
        }

        auto merge_start = chrono::high_resolution_clock::now();
        LevelInfo next_level = merge_customers(all_levels[L], s, truck_times, drone_times);
        //cout << "Nodes in next_level: ";
        //for (const auto& node : next_level.nodes) cout << node.id << " ";
        //cout << endl;
        auto merge_end = chrono::high_resolution_clock::now();
        double merge_time = chrono::duration<double>(merge_end - merge_start).count();
        int reduction = all_levels[L].nodes.size() - next_level.nodes.size();
        if (reduction < 1) {
            break;
        }
        all_levels.push_back(next_level);
        NEO_LEVELS_USED = int(all_levels.size());

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
        update_node_index_cache(next_level);
        evaluate_solution(s, &next_level);
        neo::check(abs(s.makespan - s_current.makespan) < EPSILON, "CVRP contraction changed objective");
        cvrp_verify_solution(s, &next_level);
        
        L++;
        
        (void)level_time;
        (void)merge_time;
        (void)project_time;
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
        (void)unmerge_time;
        truck_times = all_levels[prev_level_id].truck_time_matrix;
        drone_times = all_levels[prev_level_id].drone_time_matrix;

        C1 = all_levels[prev_level_id].C1_level;
        C2 = all_levels[prev_level_id].C2_level;
        num_nodes = all_levels[prev_level_id].nodes.size();
        
        /*cout << "Level " << prev_level_id << " stats:" << endl;
        cout << "  Nodes: " << num_nodes << endl;
        cout << "  C1: " << C1.size() << ", C2: " << C2.size() << endl;
        cout << "  Matrix: " << truck_times.size() << "x" 
             << (truck_times.empty() ? 0 : truck_times[0].size()) << endl;*/

        update_node_index_cache(all_levels[prev_level_id]);
        // CASE 1: LEVEL 0 - DÙNG EVALUATE VÀ TABU KHÔNG CÓ LEVEL
        if (prev_level_id == 0) {
            // Final refinement logging removed for performance.
            merged_nodes_info.clear();
            internal_distance_cache.clear();
            
            evaluate_solution(s, nullptr);
            
            auto refine_start = chrono::high_resolution_clock::now();
            
            s = tabu_search(s, nullptr);
            
            evaluate_solution(s, nullptr);
            
            auto refine_end = chrono::high_resolution_clock::now();
            double refine_time = chrono::duration<double>(refine_end - refine_start).count();
            (void)refine_time;
            best_overall = s;
        }
        // CASE 2: LEVEL 1, 2, 3... - VẪN DÙNG MULTILEVEL
        else {
            // CLEAR MERGED INFO CỦA LEVEL CAO HƠN
            auto it = merged_nodes_info.begin();
            while (it != merged_nodes_info.end()) {
                if (it->second.level_id > prev_level_id) {
                    it = merged_nodes_info.erase(it);
                } else {
                    ++it;
                }
            }
            
            evaluate_solution(s, &all_levels[prev_level_id]);
            
            auto refine_start = chrono::high_resolution_clock::now();
            s = tabu_search(s, &all_levels[prev_level_id]);
            evaluate_solution(s, &all_levels[prev_level_id]);
            
            auto refine_end = chrono::high_resolution_clock::now();
            double refine_time = chrono::duration<double>(refine_end - refine_start).count();
            (void)refine_time;
            (void)unmerge_time;
            best_overall = s;
        }
    }

    cvrp_verify_solution(best_overall, nullptr);
    return best_overall;
}


// CLI/batch comparison adapter; it does not replace any search phase above.
int run_neo_cli(int argc,char **argv,bool use_multilevel) {
    try {
        namespace fs=std::filesystem;
        fs::path input="NEO_CVRP_62_pairs",csv_path,solution_dir;
        int runs=1,max_iter_arg=0;double iter_k_arg=16;unsigned seed=4321;bool check_only=false;
        long long budget_arg=-1;
        for(int i=1;i<argc;++i) {
            string key=argv[i];
            if(key=="--check-only"){check_only=true;continue;}
            if(key=="--verify-moves"){NEO_VERIFY=true;continue;}
            if(key=="--help" || key=="-h") {
                cout<<(use_multilevel ? "ML_NEO_CVRP" : "Tabu_Neo")<<" [instance.vrp | dataset-folder] [options]\n"
                    <<"--csv file --solution-dir folder --runs N --seed N\n"
                    <<"--max_levels N --merge_ratio F --iter_k K --max_iter N\n"
                    <<"--budget B (total iteration ceiling per run, both variants)\n"
                    <<"Defaults: seed=4321, iter_k=16, max_levels=4; auto budget=(2*max_levels+1)*phase_iterations.\n"
                    <<"For Tabu_Neo, max_levels is the matched ML depth; no coarsening is performed.\n"
                    <<"--tabu_factor F --tabu1 F --tabu2 F --delta1 F --delta2 F --delta3 F --delta4 F\n"
                    <<"--check-only --verify-moves\n";
                return 0;
            }
            if(key.rfind("--",0)!=0){input=key;continue;}
            neo::check(i+1<argc,"Missing value for "+key);string val=argv[++i];
            if(key=="--instance")input=val;
            else if(key=="--csv")csv_path=val;
            else if(key=="--solution-dir")solution_dir=val;
            else if(key=="--runs")runs=stoi(val);
            else if(key=="--seed")seed=static_cast<unsigned>(stoul(val));
            else if(key=="--max_levels")MAX_LEVELS=stoi(val);
            else if(key=="--merge_ratio"){MERGE_RATIO=stod(val);if(MERGE_RATIO>1)MERGE_RATIO/=100;}
            else if(key=="--iter_k")iter_k_arg=stod(val);
            else if(key=="--max_iter")max_iter_arg=stoi(val);
            else if(key=="--budget")budget_arg=stoll(val);
            else if(key=="--tabu_factor")TABU_TENURE_FACTOR=stod(val);
            else if(key=="--tabu1")tabu1=stod(val);
            else if(key=="--tabu2")tabu2=stod(val);
            else if(key=="--delta1")delta1=stod(val);
            else if(key=="--delta2")delta2=stod(val);
            else if(key=="--delta3")delta3=stod(val);
            else if(key=="--delta4")delta4=stod(val);
            else throw runtime_error("Unknown flag: "+key);
        }
        neo::check(runs>0 && MAX_LEVELS>=0 && MAX_LEVELS<=100 && max_iter_arg>=0 && budget_arg>=-1,"Invalid integer parameters");
        neo::check(isfinite(iter_k_arg) && iter_k_arg>=0,"Invalid iter_k");
        neo::check(isfinite(MERGE_RATIO) && MERGE_RATIO>0 && MERGE_RATIO<1,"Require 0 < merge_ratio < 1");
        neo::check(isfinite(tabu1) && isfinite(tabu2) && tabu1>0 && tabu2>=tabu1,"Require 0 < tabu1 <= tabu2");
        neo::check(isfinite(TABU_TENURE_FACTOR) && TABU_TENURE_FACTOR>=0,"Invalid tabu_factor");
        for(double d:{delta1,delta2,delta3,delta4})neo::check(isfinite(d) && d>=0 && d<=1,"Invalid delta");
        vector<fs::path> files;
        if(fs::is_regular_file(input))files.push_back(input);
        else {
            neo::check(fs::is_directory(input),"Input directory not found");
            for(const auto &entry:fs::recursive_directory_iterator(input))
                if(entry.is_regular_file() && neo::upper(entry.path().extension().string())==".VRP")files.push_back(entry.path());
        }
        sort(files.begin(),files.end());neo::check(!files.empty(),"No VRP instances found");
        ofstream csv;
        if(!csv_path.empty()) {
            if(!csv_path.parent_path().empty())fs::create_directories(csv_path.parent_path());
            csv.open(csv_path);
            neo::check(bool(csv),"Cannot open CSV");
            csv<<"instance,dataset,run,seed,customers,vehicles_limit,vehicles_used,capacity,cost,feasible,runtime_s,levels_used,published_cost,reference_recomputed,reference_status,gap_pct,reason,max_levels,merge_ratio,max_iter,tabu_factor,tabu1,tabu2,delta1,delta2,delta3,delta4,iter_k,solution,algorithm,phase_max_iter,iteration_budget,iterations_used,tabu_calls\n";
        }
        if(!solution_dir.empty())fs::create_directories(solution_dir);
        int failures=0,issues=0;
        for(const auto &file:files) {
            try {
                const auto problem=neo::read(file);const auto reference=neo::reference(problem);
                if(reference.status!="PASS")++issues;
                cout<<problem.name<<" reference="<<reference.status<<endl;
                for(int run=0;run<(check_only?1:runs);++run) {
                    neo::Validation validation;neo::Routes routes;
                    double seconds=0;int levels=0;unsigned run_seed=seed+static_cast<unsigned>(run);
                    int phase_max_iter=0;
                    NEO_ITERATIONS_USED=0;NEO_TABU_CALLS=0;NEO_ITERATION_BUDGET=-1;
                    if(!check_only) {
                        read_dataset(file.string());internal_distance_cache.clear();node_id_to_index_cache.clear();
                        weights.assign(6,1.0);scorePi.assign(6,0.0);used_count.assign(6,0.0);NEO_LEVELS_USED=1;
                        if(iter_k_arg>0)MAX_ITER=max(1,int(llround(iter_k_arg*(problem.n-1))));
                        if(max_iter_arg>0)MAX_ITER=max_iter_arg;
                        phase_max_iter=MAX_ITER;
                        NEO_ITERATION_BUDGET=budget_arg>=0 ? budget_arg : 1LL*phase_max_iter*(2*MAX_LEVELS+1);
                        neo::check(NEO_ITERATION_BUDGET<=numeric_limits<int>::max(),"Iteration budget exceeds supported integer range");
                        if(!use_multilevel)MAX_ITER=static_cast<int>(NEO_ITERATION_BUDGET);
                        srand(run_seed);
                        auto start=chrono::steady_clock::now();
                        Solution solution;
                        if(use_multilevel)solution=multilevel_tabu_search();
                        else {
                            solution=init_greedy_solution();
                            solution=tabu_search(solution,nullptr);
                            cvrp_verify_solution(solution,nullptr);
                        }
                        seconds=chrono::duration<double>(chrono::steady_clock::now()-start).count();levels=NEO_LEVELS_USED;
                        routes=cvrp_original_routes(solution,nullptr);validation=neo::validate(problem,routes);
                        neo::check(abs(validation.cost-solution.makespan)<EPSILON,"Final cost mismatch");
                        if(!validation.feasible)++failures;
                        cout<<"  run="<<run+1<<" seed="<<run_seed<<" cost="<<validation.cost<<" feasible="<<(validation.feasible?"YES":"NO")<<" seconds="<<seconds<<endl;
                        cout<<"  algorithm="<<(use_multilevel?"ML":"TABU")<<" phase_max_iter="<<phase_max_iter
                            <<" iteration_budget="<<NEO_ITERATION_BUDGET<<" iterations_used="<<NEO_ITERATIONS_USED
                            <<" tabu_calls="<<NEO_TABU_CALLS<<endl;
                        if(files.size()==1)print_solution(solution);
                        if(validation.feasible && !solution_dir.empty()) {
                            ofstream out(solution_dir/(problem.name+"_seed"+to_string(run_seed)+".sol"));neo::check(bool(out),"Cannot write solution");
                            int route=0;for(const auto &r:routes){out<<"Route #"<<++route<<":";for(int node:r)out<<' '<<node;out<<'\n';}
                            out<<"Cost "<<fixed<<setprecision(0)<<validation.cost<<'\n';
                        }
                    }
                    if(csv) {
                        csv<<neo::quote(problem.name)<<','<<neo::quote(file.parent_path().parent_path().filename().string())<<','<<run+1<<','<<run_seed<<','<<problem.n-1<<','<<problem.k<<',';
                        if(!check_only)csv<<validation.loads.size();
                        csv<<','<<problem.capacity<<',';
                        if(!check_only)csv<<fixed<<setprecision(0)<<validation.cost;
                        csv<<','<<(check_only?"":validation.feasible?"YES":"NO")<<','<<fixed<<setprecision(6)<<seconds<<','<<levels<<',';
                        if(isfinite(reference.published))csv<<reference.published;
                        csv<<',';
                        if(reference.valid)csv<<reference.recomputed;
                        csv<<','<<reference.status<<',';
                        if(!check_only && validation.feasible && reference.status=="PASS" && reference.published>0)csv<<100*(validation.cost-reference.published)/reference.published;
                        csv<<','<<neo::quote(validation.reason)<<','<<MAX_LEVELS<<','<<MERGE_RATIO<<','<<MAX_ITER<<','<<TABU_TENURE_FACTOR<<','<<tabu1<<','<<tabu2<<','<<delta1<<','<<delta2<<','<<delta3<<','<<delta4<<','<<iter_k_arg<<',';
                        if(!check_only)csv<<neo::quote(neo::solution_json(routes));
                        csv<<','<<(use_multilevel?"ML":"TABU")<<','<<phase_max_iter<<',';
                        if(!check_only)csv<<NEO_ITERATION_BUDGET;
                        csv<<','<<NEO_ITERATIONS_USED<<','<<NEO_TABU_CALLS<<'\n';csv.flush();
                    }
                }
            } catch(const exception &e){++failures;cerr<<file.string()<<": "<<e.what()<<'\n';}
        }
        cout<<"SUMMARY instances="<<files.size()<<" failures="<<failures<<" reference_issues="<<issues<<'\n';return failures?2:0;
    }catch(const exception &e){cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}

#ifndef NEO_CVRP_NO_MAIN
int main(int argc,char **argv) {
    return run_neo_cli(argc,argv,true);
}
#endif
