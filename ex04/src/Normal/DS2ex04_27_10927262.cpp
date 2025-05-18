/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex03_27_10927262.cpp
 * @brief A program that implements hash tables with linear probing and double hashing to efficiently manage and organize graduate student data.
 * @version 1.5.0
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
#include <utility>
#include <vector>      // Dynamic array container
#include <queue>
#include <deque>
#include <condition_variable>
#include <functional>
#include <unordered_set>
#include <chrono>
#include <future>
#include <random>

// C Standard Library (C++ wrapper headers, alphabetical order)
#include <cstddef>     // Fundamental types (size_t, nullptr_t)
#include <cstdlib>     // General utilities (malloc, exit, atoi)
#include <cstring>     // C-style string operations (strcpy, memcmp)

#define MAX_LEN 12    // Array size of student id and name.
#define FAST_IO() \
    std::ios_base::sync_with_stdio(false); \
    std::cin.tie(nullptr); \
    std::cout.tie(nullptr);

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
    explicit ThreadPool(size_t thread_count)
        : stop(false), threads(thread_count), thread_exec_times(thread_count) {
        for (size_t i = 0; i < thread_count; ++i) {
            queues.emplace_back(std::make_unique<WorkerQueue>());
        }
        for (size_t i = 0; i < thread_count; ++i) {
            threads[i] = std::thread([this, i] { WorkerLoop(i); });
        }
    }

    ~ThreadPool() {
        stop = true;
        for (auto& queue : queues) {
            std::lock_guard<std::mutex> lock(queue->mtx);
            queue->cv.notify_all();
        }
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        #ifdef DEBUG
            for (size_t i = 0; i < thread_exec_times.size(); ++i) {
                auto nanos = std::chrono::nanoseconds(thread_exec_times[i].load());
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(nanos);
                std::cout << "Thread " << i << " total task execution time: "
                        << millis.count() << " ms" << std::endl;
            }
        #endif
    }

    template <class F, class... Args>
    auto Enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using ReturnType = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::function<void()> wrapped = [this, task]() {
            (*task)();
            if (--active_tasks == 0) {
                std::unique_lock<std::mutex> lock(wait_mutex);
                wait_cv.notify_all();
            }
        };

        active_tasks++;  // 新任務入列

        size_t index = dist(rng) % queues.size();
        {
            std::lock_guard<std::mutex> lock(queues[index]->mtx);
            queues[index]->tasks.emplace_front(std::move(wrapped));
        }
        queues[index]->cv.notify_one();
        return task->get_future();
    }


    void WaitAll() {
        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_cv.wait(lock, [this]() { return active_tasks == 0; });
    }

 private:
    struct WorkerQueue {
        std::deque<std::function<void()>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
    };

    void WorkerLoop(size_t index) {
        using clock = std::chrono::steady_clock;
        while (!stop) {
            std::function<void()> task;
            if (PopTask(index, task) || StealTask(index, task)) {
                auto start = clock::now();
                task();
                auto end = clock::now();

                // 傳入整數型態給 fetch_add
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                thread_exec_times[index].fetch_add(duration);
            } else {
                std::unique_lock<std::mutex> lock(queues[index]->mtx);
                queues[index]->cv.wait_for(lock, std::chrono::milliseconds(100));
            }
        }
    }

    bool PopTask(size_t index, std::function<void()>& task) {
        std::lock_guard<std::mutex> lock(queues[index]->mtx);
        if (!queues[index]->tasks.empty()) {
            task = std::move(queues[index]->tasks.front());
            queues[index]->tasks.pop_front();
            return true;
        }
        return false;
    }

    bool StealTask(size_t thief_index, std::function<void()>& task) {
        size_t n = queues.size();
        for (size_t i = 0; i < n; ++i) {
            size_t victim = (thief_index + i + 1) % n;
            std::lock_guard<std::mutex> lock(queues[victim]->mtx);
            if (!queues[victim]->tasks.empty()) {
                task = std::move(queues[victim]->tasks.back());
                queues[victim]->tasks.pop_back();
                return true;
            }
        }
        return false;
    }

    std::atomic<bool> stop;
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<WorkerQueue>> queues;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist;

    std::atomic<size_t> active_tasks{0};
    std::mutex wait_mutex;
    std::condition_variable wait_cv;

    std::vector<std::atomic<int64_t>> thread_exec_times;
};

template <typename NodeType>
class DirectedGraph {
 public:
    // Add a directed edge: publisher -> subscriber with given weight
    void AddEdge(const NodeType& publisher, const NodeType& subscriber, float weight) {
        // 確保 publisher 存在
        adj_list[publisher].emplace_back(subscriber, weight);
        ++node_count;

        // 如果 publisher 是新的，更新 publisher_list
        if (adj_list[publisher].size() == 1) {
            ++publisher_count;
            publisher_list.emplace_back(publisher, 0);
        }

        // 確保 subscriber 也作為 key 存在於 adj_list，但值為空
        if (adj_list.find(subscriber) == adj_list.end()) {
            adj_list[subscriber] = {}; // 插入空 vector
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

        for (const auto& pair : publisher_list) {
            const NodeType& publisher = pair.first;
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
        buffer << "<<< There are " << publisher_list.size() << " IDs in total. >>>\n";

        for (size_t i = 0; i < publisher_list.size(); ++i) {
            const NodeType& key = publisher_list[i].first;

            auto it = reachable_vec.find(key);
            if (it != reachable_vec.end()) {
                std::vector<std::vector<NodeType>>& all_paths = it->second;

                // 針對 reachable_vec[key] 的每個 path 進行排序
                for (auto& path : all_paths) {
                    std::stable_sort(path.begin(), path.end(),
                        [](const NodeType& a, const NodeType& b) {
                            return a < b;
                        });
                }

                buffer << "[" << std::setw(3) << i + 1 << "] " << key << "(" << publisher_list[i].second << "): \n";

                // 遍歷所有的 path
                size_t path_idx = 1;
                for (const auto& path : all_paths) {
                    size_t line_counter = 1;
                    
                    for (size_t j = 0; j < path.size(); ++j) {
                        buffer << "\t(" << std::setw(2) << j + 1 << ") " << path[j];

                        if (line_counter++ == 12) {
                            buffer << '\n';
                            line_counter = 1;
                        }
                    }

                    buffer << '\n';
                }
            }
        }

        cnt_output << buffer.str();
    }

    void ComputeAllConnectionCounts(const std::string& mode, size_t min_batch_size = 4) {
        #ifdef DEBUG
            auto start_time = std::chrono::steady_clock::now();
            DEBUG_LOG("Starting ComputeAllConnectionCounts with " 
                    << std::thread::hardware_concurrency() << " threads, min_batch_size=" << min_batch_size);
        #endif

        std::unordered_map<NodeType, std::vector<NodeType>> local_reachables;
        std::unordered_map<NodeType, size_t> local_counts;
        std::mutex result_mutex;

        {
            ThreadPool pool(std::thread::hardware_concurrency());
            size_t total_keys = publisher_list.size();
            size_t batch_count = (total_keys + min_batch_size - 1) / min_batch_size;

            for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
                size_t start_index = batch_index * min_batch_size;
                size_t end_index = std::min(start_index + min_batch_size, total_keys);

                pool.Enqueue([&, start_index, end_index]() {
                    // 本任務用的暫存結果
                    std::unordered_map<NodeType, std::vector<NodeType>> batch_reachables;
                    std::unordered_map<NodeType, size_t> batch_counts;

                    for (size_t i = start_index; i < end_index; ++i) {
                        NodeType key = publisher_list[i].first;
                        std::vector<NodeType> visited_vec = RunSimpleBFS(key);

                        batch_reachables[key] = std::move(visited_vec);
                        batch_counts[key] = batch_reachables[key].size();
                    }

                    // 合併到全局結果
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        for (auto& pair : batch_reachables) {
                            local_reachables[pair.first] = std::move(pair.second);
                        }

                        for (auto& pair : batch_counts) {
                            local_counts[pair.first] = pair.second;
                        }
                    }
                });
            }

            pool.WaitAll();
        }

        // 將結果放回全局 reachable_vec 與 publisher_list
        for (const auto& pair : local_reachables) {
            reachable_vec[pair.first] = { pair.second };
        }

        for (auto& entry : publisher_list) {
            if (local_counts.count(entry.first)) {
                entry.second = local_counts[entry.first];
            }
        }

        std::sort(publisher_list.begin(), publisher_list.end(),
            [](const std::pair<NodeType, size_t>& a, const std::pair<NodeType, size_t>& b) {
                return a.second != b.second ? a.second > b.second : a.first < b.first;
            });

        #ifdef DEBUG
            auto end_time = std::chrono::steady_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            DEBUG_LOG("ComputeAllConnectionCounts completed in " << total_duration.count() << "ms");
            DEBUG_LOG("Processed " << publisher_list.size() << " nodes");
        #endif
    }

    bool Empty() const {
        return adj_list.empty();
    }

    void Clear() {
        adj_list.clear();
        reachable_vec.clear();
        publisher_list.clear();
        publisher_count = 0;
        node_count = 0;
    }

    void Graph() {
        adj_list.max_load_factor(0.25);
        adj_list.reserve(2048);
        reachable_vec.max_load_factor(0.25);
        reachable_vec.reserve(2048);
        publisher_list.reserve(2048);
    }

    size_t GetPubCount() const {
        return publisher_count;
    }

    size_t GetnNodeCount() const {
        return node_count;
    }

    size_t GetReachCount() const {
        return publisher_list.size();
    }

 private:
    mutable std::unordered_map<NodeType, std::vector<std::pair<NodeType, float>>> adj_list;
    mutable std::unordered_map<NodeType, std::vector<std::vector<NodeType>>> reachable_vec;
    mutable std::vector<std::pair<NodeType, size_t>> publisher_list;
    mutable std::mutex log_mtx; 
    mutable std::mutex bfs_write_mutex; 

    size_t publisher_count = 0;
    size_t node_count = 0;

    // Sorts the keys only once unless the graph changes
    void SortKeys() const {
        // **直接排序**
        if (publisher_list.size() < 10000) {
            std::sort(publisher_list.begin(), publisher_list.end(), 
                [](const auto& a, const auto& b) { return a.first < b.first; });
        } else {
            // **手動分割**
            auto mid = publisher_list.begin() + publisher_list.size() / 2;
            std::vector<std::pair<NodeType, size_t>> left(publisher_list.begin(), mid);
            std::vector<std::pair<NodeType, size_t>> right(mid, publisher_list.end());

            std::sort(left.begin(), left.end(), 
                [](const auto& a, const auto& b) { return a.first < b.first; });

            std::sort(right.begin(), right.end(), 
                [](const auto& a, const auto& b) { return a.first < b.first; });

            std::merge(left.begin(), left.end(), right.begin(), right.end(), publisher_list.begin(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }
    }

    std::vector<NodeType> RunSimpleBFS(const NodeType& source_node) {
        std::queue<NodeType> queue;
        std::unordered_set<NodeType> visited;

        queue.push(source_node);
        visited.insert(source_node);

        while (!queue.empty()) {
            NodeType current = queue.front();
            queue.pop();

            auto it = adj_list.find(current);
            if (it == adj_list.end()) continue;

            const auto& neighbors = it->second;
            for (const auto& neighbor_pair : neighbors) {
                const NodeType& neighbor = neighbor_pair.first;
                if (visited.insert(neighbor).second) {
                    queue.push(neighbor);
                }
            }
        }

        visited.erase(source_node);
        return std::vector<NodeType>(visited.begin(), visited.end());
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
    FAST_IO();

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
