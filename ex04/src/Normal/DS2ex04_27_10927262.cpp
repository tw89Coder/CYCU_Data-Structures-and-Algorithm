/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex03_27_10927262.cpp
 * @brief A program that implements hash tables with linear probing and double hashing to efficiently manage and organize graduate student data.
 * @version 1.1.0
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
#include <numeric>     // Numeric operations (accumulate, inner_product)
#include <sstream>     // String streams (istringstream, ostringstream)
#include <stdexcept>   // Standard exceptions (runtime_error, logic_error)
#include <string>      // String class and operations
#include <thread>      // Thread support (std::thread)
#include <vector>      // Dynamic array container
#include <queue>
#include <unordered_set>
#include <condition_variable>
#include <functional>

// C Standard Library (C++ wrapper headers, alphabetical order)
#include <cstddef>     // Fundamental types (size_t, nullptr_t)
#include <cstdint>     // Fixed-width integer types (int32_t, uint64_t)
#include <cstdlib>     // General utilities (malloc, exit, atoi)
#include <cstring>     // C-style string operations (strcpy, memcmp)

#define COLUMNS 6     // Number of scores for each student.
#define MAX_LEN 12    // Array size of student id and name.
#define BIG_INT 255   // Integer upper bound.

#ifdef DEBUG
    // Thread-safe debug logging
    #define DEBUG_LOG(msg) { std::lock_guard<std::mutex> lock(log_mtx); \
                             std::cout << "[DEBUG] " << msg << '\n'; }
#else
    // Expands to nothing in non-DEBUG builds (no runtime overhead) 
    #define DEBUG_LOG(msg)   
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
    explicit ThreadPool(size_t threads) : stop(false), busy_threads(0) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        if (stop && tasks.empty()) return;

                        task = std::move(tasks.front());
                        tasks.pop();
                        busy_threads++;
                    }

                    task();
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        busy_threads--;
                        cv_finished.notify_one();
                    }
                }
            });
        }
    }

    template<class F>
    void Enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace([this, func = std::forward<F>(f)] {
                func();
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    busy_threads--;
                    cv_finished.notify_one();
                }
            });
            busy_threads++;
        }
        condition.notify_one();
    }

    void WaitAll() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_finished.wait(lock, [this] { return tasks.empty() && (busy_threads == 0); });
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
    std::condition_variable condition;
    std::condition_variable cv_finished;

    std::atomic<bool> stop;
    std::atomic<size_t> busy_threads;
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
        } else {
            sorted_edges_valid = false;
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
        SortAllEdges();

        std::ostringstream buffer;

        buffer << "<<< There are " << publisher_count << " IDs in total. >>>\n";

        for (const NodeType& publisher : sorted_keys) {
            const std::vector<std::pair<NodeType, float>>& sorted_edges = adj_list_sorted.at(publisher);

            if (sorted_edges.empty()) continue;

            ++publisher_index;
            buffer << "[" << std::setw(3) << publisher_index << "] " << publisher << ": \n";

            for (size_t i = 0; i < sorted_edges.size(); ++i) {
                const std::pair<NodeType, float>& edge = sorted_edges[i];
                buffer << "\t(" << std::setw(2) << i + 1 << ") " << edge.first << "," << std::setw(7) << edge.second;

                if ((i + 1) % 12 == 0) buffer << '\n';
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

        for (typename std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>>::iterator it = reachable_map.begin();
            it != reachable_map.end(); ++it) {
            std::vector<std::tuple<NodeType, float, float>>& vec = it->second;

            std::stable_sort(vec.begin(), vec.end(),
                [](const std::tuple<NodeType, float, float>& a, const std::tuple<NodeType, float, float>& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });
        }

        for (size_t i = 0; i < reachable_counts.size(); ++i) {
            const NodeType& key = reachable_counts[i].first;

            typename std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>>::const_iterator it = reachable_map.find(key);
            if (it == reachable_map.end()) continue;

            const std::vector<std::tuple<NodeType, float, float>>& reachables = it->second;
            if (reachables.empty()) continue;

            buffer << "[" << std::setw(3) << i + 1 << "] " << key << "(" << reachable_counts[i].second << "): \n";

            for (size_t j = 0; j < reachables.size(); ++j) {
                const std::tuple<NodeType, float, float>& tup = reachables[j];
                buffer << "\t(" << std::setw(2) << j + 1 << ") " 
                       << std::get<0>(tup);

                if ((j + 1) % 12 == 0) {
                    buffer << '\n';
                }
            }

            buffer << '\n';
        }

        cnt_output << buffer.str();
    }

    void ComputeAllConnectionCounts(const std::string& mode) {
        reachable_map.clear();
        reachable_counts.clear();
        SortKeys();

        {
            ThreadPool pool(std::thread::hardware_concurrency());
            std::mutex results_mutex;
            
            for (const auto& key : sorted_keys) {
                pool.Enqueue([&, key]() {
                    std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>> local_map;
                    std::pair<NodeType, size_t> local_count;  // 改為單個 pair
                    
                    BFSUpdateReachable(key, mode, local_map, local_count);
                    
                    std::lock_guard<std::mutex> lock(results_mutex);
                    for (auto& pair : local_map) {
                        reachable_map[pair.first] = std::move(pair.second);
                    }
                    reachable_counts.push_back(local_count);  // 改為 push_back
                });
            }
            
            // ThreadPool 析構會自動等待所有任務完成
            pool.WaitAll();
        }
        
        std::stable_sort(reachable_counts.begin(), reachable_counts.end(),
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
        adj_list_sorted.clear();
        reachable_map.clear();
        reachable_counts.resize(0);
        is_sorted = false;
        sorted_edges_valid = false;
        publisher_count = 0;
        node_count = 0;
    }

    void Graph() {
        adj_list.max_load_factor(0.25);
        adj_list_sorted.max_load_factor(0.25);
        reachable_map.max_load_factor(0.25);
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
    std::unordered_map<NodeType, std::vector<std::pair<NodeType, float>>> adj_list;

    mutable std::vector<NodeType> sorted_keys;
    mutable std::unordered_map<NodeType, std::vector<std::pair<NodeType, float>>> adj_list_sorted;
    mutable std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>> reachable_map;
    mutable std::vector<std::pair<NodeType, size_t>> reachable_counts;
    mutable bool is_sorted = false;
    mutable bool sorted_edges_valid = false;

    size_t publisher_count = 0;
    size_t node_count = 0;

    std::mutex mtx; 

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

    void SortAllEdges() const {
        if (sorted_edges_valid) return;

        adj_list_sorted.clear();

        for (typename std::unordered_map<NodeType, std::vector<std::pair<NodeType, float>>>::const_iterator it = adj_list.begin(); it != adj_list.end(); ++it) {
            const NodeType& node = it->first;
            const std::vector<std::pair<NodeType, float>>& edges = it->second;

            std::vector<std::pair<NodeType, float>> sorted_edges(edges.begin(), edges.end());

            if (sorted_edges.size() < 32) {
                std::stable_sort(sorted_edges.begin(), sorted_edges.end(),
                    [](const std::pair<NodeType, float>& first, const std::pair<NodeType, float>& second) {
                        return first.first < second.first;
                    });
            } else {
                std::sort(sorted_edges.begin(), sorted_edges.end(),
                    [](const std::pair<NodeType, float>& first, const std::pair<NodeType, float>& second) {
                        return first.first < second.first;
                    });
            }

            adj_list_sorted[node] = std::move(sorted_edges);
        }

        sorted_edges_valid = true;
    }

    void BFSUpdateReachable(const NodeType& key, const std::string& mode,
                            std::unordered_map<NodeType, std::vector<std::tuple<NodeType, float, float>>>& local_reachable_map,
                            std::pair<NodeType, size_t>& local_count) const {
        std::unordered_map<NodeType, float> weight_map;
        std::unordered_map<NodeType, float> edge_map;
        std::unordered_set<NodeType> visited;
        std::queue<NodeType> q;

        // 預先分配記憶體來減少 rehash
        weight_map.reserve(adj_list.size());
        edge_map.reserve(adj_list.size());
        visited.reserve(adj_list.size());

        weight_map[key] = 0.0f;
        edge_map[key] = 0.0f;

        auto it = adj_list.find(key);
        if (it != adj_list.end()) {
            for (const auto& [neighbor, edge_weight] : it->second) {
                weight_map[neighbor] = edge_weight;
                edge_map[neighbor] = edge_weight;
                q.push(neighbor);
                visited.insert(neighbor);
            }
        }

        while (!q.empty()) {
            NodeType current = q.front();
            q.pop();
            float current_weight = weight_map[current];

            auto neighbor_it = adj_list.find(current);
            if (neighbor_it == adj_list.end()) continue;

            for (const auto& [neighbor, edge_weight] : neighbor_it->second) {
                if (neighbor == key) continue;

                float total_weight = current_weight + edge_weight;

                auto weight_it = weight_map.find(neighbor);
                bool should_update = (weight_it == weight_map.end()) ||
                                    (mode == "min" && total_weight < weight_it->second) ||
                                    (mode == "max" && total_weight > weight_it->second);

                if (should_update) {
                    weight_map[neighbor] = total_weight;
                    edge_map[neighbor] = edge_weight;
                    
                    if (visited.insert(neighbor).second) {  // 使用 insert() 的回傳值來減少查找次數
                        q.push(neighbor);
                    }
                }
            }
        }

        std::vector<std::tuple<NodeType, float, float>> result;
        result.reserve(weight_map.size());

        for (const auto& [node, acc_weight] : weight_map) {
            if (node == key) continue;
            result.emplace_back(node, edge_map.at(node), acc_weight);
        }

        local_reachable_map[key] = std::move(result);
        local_count = {key, weight_map.size() - 1};
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
