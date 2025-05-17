/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex03_27_10927262.cpp
 * @brief A program that implements hash tables with linear probing and double hashing to efficiently manage and organize graduate student data.
 * @version 1.3.0
 *
 * @details
 * This program implements two different hashing techniques, linear probing and double hashing, for storing and managing graduate student data. 
 * It supports insertion and search operations in both hash table variants, ensuring efficient data retrieval and organization. 
 * The program is designed to demonstrate the use of hashing techniques in managing dynamic datasets like student records.
 * The user can perform operations such as building hash tables, and analyzing search performance metrics (successful and unsuccessful search comparisons).
 *
 * @author 
 * - Group 27
 * - 10927262 呂易鴻
 */

// C++ Standard Library (alphabetical order)
#include <algorithm>   // Algorithms (sort, find, transform)
#include <atomic>      // Atomic operations and thread synchronization
#include <fstream>     // File stream operations (ifstream, ofstream)
#include <iomanip>     // I/O formatting (setw, setprecision)
#include <iostream>    // Standard I/O streams (cin, cout)
#include <limits>      // Numeric limits (numeric_limits)
#include <unordered_map>
#include <sys/stat.h>
#include <mutex>       // Mutual exclusion (mutex, lock_guard)
#include <sstream>     // String streams (istringstream, ostringstream)
#include <string>      // String class and operations
#include <thread>      // Thread support (std::thread)
#include <vector>      // Dynamic array container
#include <queue>
#include <deque>
#include <condition_variable>
#include <functional>
#include <chrono>

// C Standard Library (C++ wrapper headers, alphabetical order)
#include <cstddef>     // Fundamental types (size_t, nullptr_t)
#include <cstdlib>     // General utilities (malloc, exit, atoi)
#include <cstring>     // C-style string operations (strcpy, memcmp)

#define COLUMNS 6     // Number of scores for each student.
#define MAX_LEN 12    // Array size of student id and name.
#define BIG_INT 255   // Integer upper bound.

#ifdef DEBUG
    // Thread-safe debug logging
    #define THREAD_LOG(msg) { std::lock_guard<std::mutex> lock(log_mtx); \
                             std::cout << "[THREAD] " << msg << '\n'; }
#else
    // Expands to nothing in non-DEBUG builds (no runtime overhead) 
    #define THREAD_LOG(msg)   
#endif

/**
 * @struct StudentType
 * @brief Represents student record data.
 */
typedef struct st {
    char publisher[MAX_LEN];             // student id
    char subscriber[MAX_LEN];           // student name
    float weight;                    // The average of scores.
} StudentType;

class ThreadPool {
public:
    explicit ThreadPool(size_t threads, size_t batch_size = 1)
        : stop(false), busy_threads(0), batch_size(batch_size) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, batch_size] {
                THREAD_LOG("Worker thread " << i << " started");
                for (;;) {
                    std::vector<std::function<void()>> task_batch;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        THREAD_LOG("Worker thread " << i << " waiting for tasks");
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        if (stop && tasks.empty()) {
                            THREAD_LOG("Worker thread " << i << " terminating");
                            return;
                        }

                        // 批次抓取任務
                        size_t count = std::min(batch_size, tasks.size());
                        for (size_t j = 0; j < count; ++j) {
                            task_batch.push_back(std::move(tasks.front()));
                            tasks.pop();
                        }
                        busy_threads += count;
                        THREAD_LOG("Worker thread " << i << " got batch of " << count
                                    << " tasks (busy: " << busy_threads << ")");
                    }

                    // 執行任務
                    for (auto& task : task_batch) {
                        auto start = std::chrono::steady_clock::now();
                        task();
                        auto end = std::chrono::steady_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                        THREAD_LOG("Worker thread " << i << " completed task in "
                                    << duration.count() << "ms");
                    }

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        busy_threads -= task_batch.size();
                        THREAD_LOG("Worker thread " << i << " batch done (busy: " << busy_threads << ")");
                        cv_finished.notify_all();
                    }
                }
            });
        }
    }

    template<class F>
    void Enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    void WaitAll() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_finished.wait(lock, [this] { return tasks.empty() && busy_threads == 0; });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    mutable std::mutex log_mtx;
    std::condition_variable condition;
    std::condition_variable cv_finished;

    std::atomic<bool> stop;
    std::atomic<size_t> busy_threads;
    size_t batch_size;
};


template <typename NodeType>
class DirectedGraph {
 public:
    // Add a directed edge: publisher -> subscriber with given weight
    void AddEdge(const NodeType& publisher, const NodeType& subscriber, float weight) {
        bool is_new_publisher = adj_list.find(publisher) == adj_list.end();
        bool is_new_subscriber = adj_list.find(subscriber) == adj_list.end();

        adj_list[publisher].emplace_back(subscriber, weight);
        ++node_count;

        if (adj_list[publisher].size() == 1) {
            ++publisher_count;
        }

        if (is_new_subscriber) {
            adj_list[subscriber] = {};
        }

        if (is_new_publisher) {
            is_sorted = false;
        }
    }

    void SaveToAdjFile(const std::string& file_number) const {
        std::string file_name = "pairs" + file_number + ".adj";
        std::ofstream adj_output(file_name.c_str());

        if (!adj_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        size_t publisher_index = 0;
        SortKeys();

        std::ostringstream buffer;
        buffer << "<<< There are " << publisher_count << " IDs in total. >>>\n";

        for (const NodeType& publisher : sorted_keys) {
            auto it = adj_list.find(publisher);
            if (it == adj_list.end() || it->second.empty()) continue;

            std::vector<std::pair<NodeType, float>>& edges = it->second;

            if (edges.size() < 32) {
                std::stable_sort(edges.begin(), edges.end(),
                    [](const auto& first, const auto& second) { return first.first < second.first; });
            } else {
                std::sort(edges.begin(), edges.end(),
                    [](const auto& first, const auto& second) { return first.first < second.first; });
            }

            ++publisher_index;
            buffer << "[" << std::setw(3) << publisher_index << "] " << publisher << ": \n";

            size_t line_counter = 1;
            for (size_t i = 0; i < edges.size(); ++i) {
                const auto& edge = edges[i];
                buffer << "\t(" << std::setw(2) << i + 1 << ") " << edge.first << "," << std::setw(7) << edge.second;
                if (line_counter++ == 12) {
                    buffer << '\n';
                    line_counter = 1;
                }
            }
            buffer << '\n';
        }

        buffer << "<<< There are " << node_count << " nodes in total. >>>\n";
        adj_output << buffer.str();
    }

    void SaveToCntFile(const std::string& file_number) const {
        std::string file_name = "pairs" + file_number + ".cnt";
        std::ofstream cnt_output(file_name.c_str());

        if (!cnt_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        std::ostringstream buffer;

        buffer << "<<< There are " << reachable_counts.size() << " IDs in total. >>>\n";

        for (const auto& key : sorted_keys) {
            std::vector<std::tuple<NodeType, float, float>>& vec = reachable_map.at(key);

            std::stable_sort(vec.begin(), vec.end(),
                [](const std::tuple<NodeType, float, float>& a, const std::tuple<NodeType, float, float>& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });
        }

        for (size_t i = 0; i < reachable_counts.size(); ++i) {
            const NodeType& key = reachable_counts[i].first;

            const std::vector<std::tuple<NodeType, float, float>>& reachables = reachable_map.at(key);

            buffer << "[" << std::setw(3) << i + 1 << "] " << key << "(" << reachable_counts[i].second << "): \n";

            size_t line_counter = 1;
            for (size_t j = 0; j < reachables.size(); ++j) {
                const std::tuple<NodeType, float, float>& tup = reachables[j];
                buffer << "\t(" << std::setw(2) << j + 1 << ") " 
                       << std::get<0>(tup);

                if (line_counter++ == 12) {
                    buffer << '\n';
                    line_counter = 1;
                }
            }

            buffer << '\n';
        }

        cnt_output << buffer.str();
    }

    void ComputeAllConnectionCounts(const std::string& mode) {
        reachable_map.clear();
        reachable_counts.clear();

        auto start_time = std::chrono::steady_clock::now();
        THREAD_LOG("Starting ComputeAllConnectionCounts with " 
                << std::thread::hardware_concurrency() << " threads");

        {
            ThreadPool pool(std::thread::hardware_concurrency());
            std::mutex results_mutex;

            const size_t mini_batch_size = 8;  // 調整這裡看效能效果
            size_t total_keys = sorted_keys.size();
            size_t task_count = 0;

            for (size_t i = 0; i < total_keys; i += mini_batch_size) {
                size_t start_idx = i;
                size_t end_idx = std::min(i + mini_batch_size, total_keys);

                task_count++;
                pool.Enqueue([&, start_idx, end_idx]() {
                    auto task_start = std::chrono::steady_clock::now();
                    THREAD_LOG("Mini batch processing keys " << start_idx << " to " << (end_idx - 1));

                    std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>> local_map;
                    std::vector<std::pair<NodeType, size_t>> local_counts;

                    for (size_t j = start_idx; j < end_idx; ++j) {
                        NodeType key = sorted_keys[j];
                        std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>> temp_map;
                        std::pair<NodeType, size_t> temp_count;
                        BFSUpdateReachable(key, mode, temp_map, temp_count);

                        for (auto& pair : temp_map) {
                            local_map[pair.first] = std::move(pair.second);
                        }
                        local_counts.push_back(std::move(temp_count));
                    }

                    {
                        std::lock_guard<std::mutex> lock(results_mutex);
                        for (auto& pair : local_map) {
                            reachable_map[pair.first] = std::move(pair.second);
                        }
                        reachable_counts.insert(reachable_counts.end(), local_counts.begin(), local_counts.end());
                    }

                    auto task_end = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(task_end - task_start);
                    THREAD_LOG("Completed mini batch [" << start_idx << ", " << end_idx - 1 << "] in " << duration.count() << "ms");
                });
            }

            THREAD_LOG("Enqueued " << task_count << " mini batch tasks for processing");
            pool.WaitAll();
        }

        auto end_time = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        THREAD_LOG("ComputeAllConnectionCounts completed in " << total_duration.count() << "ms");
        THREAD_LOG("Processed " << reachable_counts.size() << " nodes");

        std::sort(reachable_counts.begin(), reachable_counts.end(),
            [](const std::pair<NodeType, size_t>& a, const std::pair<NodeType, size_t>& b) {
                return a.second != b.second ? a.second > b.second : a.first < b.first;
            });
    }

    bool Empty() const {
        return adj_list.empty();
    }

    void Clear() {
        adj_list.clear();
        sorted_keys.resize(0);
        reachable_map.clear();
        reachable_counts.resize(0);
        is_sorted = false;
        publisher_count = 0;
        node_count = 0;
    }

    void Graph() {
        adj_list.max_load_factor(0.25);
        adj_list.reserve(1048576);
        reachable_map.max_load_factor(0.25);
        reachable_map.reserve(1048576);
    }

    size_t GetPubCount() const {
        return publisher_count;
    }

    size_t GetnNodeCount() const {
        return node_count;
    }

    size_t GetReachCount() const {
        return reachable_counts.size();
    }

 private:
    mutable std::unordered_map<NodeType, std::vector<std::pair<NodeType, float>>> adj_list;
    mutable std::vector<NodeType> sorted_keys;
    mutable std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>> reachable_map;
    mutable std::vector<std::pair<NodeType, size_t>> reachable_counts;
    mutable bool is_sorted = false;
    mutable std::mutex log_mtx; 

    size_t publisher_count = 0;
    size_t node_count = 0;

    // Sorts the keys only once unless the graph changes
    void SortKeys() const {
        if (is_sorted) return;

        sorted_keys.resize(0);
        sorted_keys.reserve(adj_list.size());

        for (const auto& pair : adj_list) {
            sorted_keys.push_back(pair.first);
        }

        if (sorted_keys.size() < 10000) {
            std::sort(sorted_keys.begin(), sorted_keys.end());
        } else {
            auto mid = sorted_keys.begin() + sorted_keys.size() / 2;
            std::vector<NodeType> left(sorted_keys.begin(), mid);
            std::vector<NodeType> right(mid, sorted_keys.end());

            std::sort(left.begin(), left.end());
            std::sort(right.begin(), right.end());

            std::merge(left.begin(), left.end(), right.begin(), right.end(), sorted_keys.begin());
        }

        is_sorted = true;
    }

    void BFSUpdateReachable(const NodeType& source_node,
                            const std::string& mode,
                            std::unordered_map<NodeType, 
                                            std::vector<std::tuple<NodeType, float, float>>>& reachable_nodes_map,
                            std::pair<NodeType, size_t>& node_statistics) const {
        // 節點數據結構
        struct NodeData {
            float edge_weight;
            float accumulated_weight;
            bool visited;
            bool used;
            
            NodeData() : 
                edge_weight(0.0f), 
                accumulated_weight(0.0f), 
                visited(false), 
                used(false) {}
                
            NodeData(float ew, float aw, bool v, bool u) :
                edge_weight(ew),
                accumulated_weight(aw),
                visited(v),
                used(u) {}
        };

        // 本地存儲（非靜態以保證線程安全）
        std::unordered_map<NodeType, int> node_to_id;
        std::vector<NodeType> id_to_node;
        std::vector<NodeData> node_data_vec;
        
        // 預留足夠空間減少重新分配
        const size_t estimated_size = adj_list.size() * 2;
        node_to_id.reserve(estimated_size);
        id_to_node.reserve(estimated_size);
        node_data_vec.resize(estimated_size);

        // 優化：預先計算模式比較結果
        const bool is_max_mode = (mode == "max");

        // 獲取或分配節點ID的lambda函數
        auto get_or_assign_id = [&](const NodeType& node) -> int {
            const auto it = node_to_id.find(node);
            if (it != node_to_id.end()) {
                return it->second;
            }

            const int new_id = static_cast<int>(id_to_node.size());
            node_to_id[node] = new_id;
            id_to_node.push_back(node);
            
            // 確保有足夠空間
            if (new_id >= static_cast<int>(node_data_vec.size())) {
                node_data_vec.resize(new_id + 1);
            }
            return new_id;
        };

        // 處理源節點
        const int src_id = get_or_assign_id(source_node);
        node_data_vec[src_id] = NodeData(0.0f, 0.0f, false, true);

        std::deque<int> traversal_queue;

        // 初始化鄰居節點
        const auto source_it = adj_list.find(source_node);
        if (source_it != adj_list.end()) {
            for (const auto& neighbor_pair : source_it->second) {
                const int neighbor_id = get_or_assign_id(neighbor_pair.first);
                node_data_vec[neighbor_id] = NodeData(
                    neighbor_pair.second, 
                    neighbor_pair.second, 
                    true,  // visited
                    true   // used
                );
                traversal_queue.push_back(neighbor_id);
            }
        }

        // BFS 主循環
        while (!traversal_queue.empty()) {
            const int curr_id = traversal_queue.front();
            traversal_queue.pop_front();

            const NodeType& current_node = id_to_node[curr_id];
            const float current_acc_weight = node_data_vec[curr_id].accumulated_weight;

            const auto it = adj_list.find(current_node);
            if (it == adj_list.end()) continue;

            // 處理所有鄰居節點
            for (const auto& neighbor_pair : it->second) {
                const NodeType& neighbor = neighbor_pair.first;
                if (neighbor == source_node) continue;

                const float edge_weight = neighbor_pair.second;
                const float total_weight = current_acc_weight + edge_weight;
                const int neighbor_id = get_or_assign_id(neighbor);

                NodeData& data = node_data_vec[neighbor_id];
                bool need_update = false;

                if (!data.used) {
                    need_update = true;
                } else if (is_max_mode && total_weight > data.accumulated_weight) {
                    need_update = true;
                }

                if (need_update) {
                    data.accumulated_weight = total_weight;
                    data.edge_weight = edge_weight;
                    if (!data.visited) {
                        data.visited = true;
                        traversal_queue.push_back(neighbor_id);
                    }
                    data.used = true;
                }
            }
        }

        // 收集可達節點
        std::vector<std::tuple<NodeType, float, float>> reachable_nodes;
        reachable_nodes.reserve(id_to_node.size() - 1);  // 減去源節點

        for (size_t i = 0; i < id_to_node.size(); ++i) {
            if (static_cast<int>(i) == src_id) continue;
            
            const NodeData& data = node_data_vec[i];
            if (!data.used) continue;

            reachable_nodes.emplace_back(
                id_to_node[i],
                data.edge_weight,
                data.accumulated_weight
            );
        }

        // 保存結果
        reachable_nodes_map[source_node] = std::move(reachable_nodes);
        node_statistics = std::make_pair(
            source_node, 
            node_data_vec[src_id].used ? id_to_node.size() - 1 : 0
        );
    }
};

/**
 * @brief Reads student records from a binary file into a vector.
 * 
 * @tparam T The type of data to read (should be StudentType).
 * @param file_name Name of the binary file to read from.
 * @param vec [out] Vector that will contain the read student records.
 */
template <typename T>
static void ReadBinary(const std::string& file_number, DirectedGraph<T>& graph) {
    std::string file_name = "pairs" + file_number + ".bin";

    std::ifstream file(file_name, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open binary file: " << file_name << "\n";
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(StudentType) != 0) {
        std::cerr << "Binary file size is not aligned with StudentType size.\n";
        return;
    }
    
    size_t count = size / sizeof(StudentType);
    std::vector<StudentType> buffer(count);

    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read binary data.\n";
        return;
    }

    file.close();

    for (const auto& student : buffer) {
        graph.AddEdge(student.publisher, student.subscriber, student.weight);
    }
}

/**
 * @brief Executes Task 1 operations using linear probing hash table.
 * 
 * Creates a hash table with linear probing collision resolution,
 * inserts all student records, calculates search performance metrics,
 * and saves results to a file.
 * 
 * @param student_info Vector containing all student records to process.
 * @param file_number Used to name the output file (e.g., "linear1.txt").
 */
static void Task1(DirectedGraph<std::string>& dir_graph, const std::string& file_number) {
    ReadBinary(file_number, dir_graph);

    dir_graph.SaveToAdjFile(file_number);

    std::cout << "<<< There are " << dir_graph.GetPubCount() << " IDs in total. >>>\n\n"
              << "<<< There are " << dir_graph.GetnNodeCount() << " nodes in total. >>>\n";
}

/**
 * @brief Executes Task 2 operations using double hashing.
 * 
 * Creates a hash table with double hashing collision resolution,
 * inserts all student records, calculates search performance metrics,
 * and saves results to a file.
 * 
 * @param student_info Vector containing all student records to process.
 * @param file_number Used to name the output file (e.g., "double1.txt").
 */
static void Task2(DirectedGraph<std::string>& dir_graph, const std::string& file_number) {
    dir_graph.ComputeAllConnectionCounts("max");
    dir_graph.SaveToCntFile(file_number);

    std::cout << "<<< There are " << dir_graph.GetReachCount() << " IDs in total. >>>\n";
}

/**
 * @brief Main program entry point.
 * 
 * Provides a menu-driven interface for hash table operations:
 * 0. Quit
 * 1. Linear probing operations
 * 2. Double hashing operations
 * 
 * Handles user input, file operations, and coordinates task execution.
 * 
 * @return int Returns 0 on normal program termination.
 */
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int select_command = 0;
    std::string file_number;
    std::string file_name;
    DirectedGraph<std::string> dir_graph;

    dir_graph.Graph();

    do {
        while (true) {
            // Display the menu options for the user
            std::cout <<
                "**** Graph data manipulation *****\n"
                "* 0. QUIT                        *\n"
                "* 1. Build adjacency lists       *\n"
                "* 2. Compute connection counts   *\n"
                "**********************************\n"
                "Input a choice(0, 1, 2): "
                << std::flush;

            std::cin >> select_command;

            // Check if the input is valid
            if (!std::cin.fail()) {
                break;
            } else {
                // Handle invalid input
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                std::cout << '\n';
                std::cout << "Command does not exist!\n";
                std::cout << '\n';
            }
        }

        // Handle the different options based on the user's choice
        switch (select_command) {
        case 0:
            break;
        case 1:
            std::cout << '\n';
            std::cout << "Input a file number ([0] Quit): " << std::flush;
            std::cin >> file_number;

            if (!dir_graph.Empty()) {
                dir_graph.Clear();
            }

            if (file_number != "0") {
                struct stat buffer;
                file_name = "pairs" + file_number + ".bin";
                std::cout << '\n';

                if (stat(file_name.c_str(), &buffer)) {
                    std::cout << '\n';
                    std::cout << "### " << file_name << " does not exist! ###\n";
                } else {
                    Task1(dir_graph, file_number);
                }
            }
            std::cout << '\n';

            break;
        case 2:
            if (dir_graph.Empty()) {
                std::cout << "### There is no graph and choose 1 first. ###\n";
            } else {
                std::cout << '\n';
                Task2(dir_graph, file_number);
            }
            std::cout << '\n';

            break;
        default:
            std::cout << '\n';
            std::cout << "Command does not exist!\n";
            std::cout << '\n';
        }
    } while (select_command != 0);  // Continue until the user selects option 0 (quit)

    return 0;
}
