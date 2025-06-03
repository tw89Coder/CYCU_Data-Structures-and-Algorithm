/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex05_27_10927262.cpp
 * @brief A program that implements a graph with weighted edges and parallel processing
 * @version 1.1.0
 *
 * @details
 * This program implements a graph data structure that supports:
 * - Adding weighted edges between nodes
 * - Finding connected components
 * - Calculating shortest paths
 * - Parallel execution of graph algorithms
 * - Saving graph data to files
 * 
 * Main features:
 * - Build adjacency lists from input data
 * - Analyze graph connectivity
 * - Compute connection counts between nodes
 * - Output results in formatted files
 *
 * @author 
 * - Group 27
 * - 10927262 呂易鴻
 */

/* System Headers */
#include <sys/stat.h>      // File system status operations (POSIX)

/* C++ Standard Library Headers */

// I/O Operations
#include <iostream>        // Standard I/O stream objects (cin, cout, cerr, clog)
#include <fstream>         // File stream operations (ifstream, ofstream)
#include <iomanip>         // Stream manipulators (setw, setprecision)
#include <sstream>         // String stream operations (istringstream, ostringstream)

// Strings and Regular Expressions
#include <string>          // String class and operations
#include <regex>           // Regular expression support (regex_match, regex_search)

// Containers
#include <vector>          // Dynamic array container
#include <deque>           // Double-ended queue container
#include <queue>           // Queue container (FIFO semantics)
#include <unordered_map>   // Hash map container (unordered associative)

// Algorithms
#include <algorithm>       // Common algorithms (sort, find, transform)
#include <numeric>         // Numeric operations (accumulate, inner_product)

// Concurrency
#include <thread>          // Thread management (std::thread)
#include <mutex>           // Mutual exclusion (mutex, lock_guard)
#include <future>          // Asynchronous operations (future, promise)
#include <functional>      // Function objects and binders

// Utilities
#include <chrono>          // Time utilities (clocks, durations)
#include <random>          // Random number generation
#include <limits>          // Numeric limits (numeric_limits)

/* C Standard Library Headers */
#include <cstddef>         // Fundamental types (size_t, nullptr_t, ptrdiff_t)

// Constant definition
constexpr size_t MAX_LEN = 12;    // Array size of student id and name.

// Input/Output optimization
inline void FAST_IO() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
}

#ifdef DEBUG
    // Thread-safe debug logging
    #define DEBUG_LOG(msg) { std::lock_guard<std::mutex> lock(log_mtx_); \
                             std::cout << "\033[36m[DEBUG]\033[0m " << msg << '\n'; }
#else
    // Expands to nothing in non-DEBUG builds (no runtime overhead)
    #define DEBUG_LOG(msg)
#endif

/**
 * @struct StudentType
 * @brief Represents student record data.
 */
typedef struct st {
    char source[MAX_LEN];
    char neighbor[MAX_LEN];
    float weight;
} StudentType;

/**
 * @brief A thread pool_ implementation with work stealing capabilities.
 *
 * Manages a pool_ of worker threads that execute enqueued tasks efficiently.
 * Features include:
 * - Per-thread task queues to reduce contention
 * - Work stealing when a thread's queue is empty
 * - Task execution time tracking
 * - Thread-safe task submission
 * - Wait mechanism for all tasks completion
 */
class ThreadPool {
 public:
    /**
     * @brief Constructs a ThreadPool with the specified number of threads.
     * 
     * Initializes worker threads and their respective task queues.
     * Each thread gets its own queue to minimize contention.
     * 
     * @param thread_count Number of worker threads to create.
     */
    explicit ThreadPool(size_t thread_count, bool enable_work_stealing = true)
        : stop(false),
          threads(thread_count),
          thread_exec_times(thread_count),
          work_stealing_enabled(enable_work_stealing) {
        // Create task queues
        for (size_t i = 0; i < thread_count; ++i) {
            queues.emplace_back(std::make_unique<WorkerQueue>());
        }

        // Launch worker threads
        for (size_t i = 0; i < thread_count; ++i) {
            threads[i] = std::thread([this, i] {
                WorkerLoop(i);
            });
        }
    }

    /**
     * @brief Destructor that safely shuts down the thread pool_.
     *
     * Signals all threads to stop, notifies them to wake up,
     * waits for completion, and in DEBUG mode prints execution
     * time statistics for each thread.
     */
    ~ThreadPool() {
        // Signal all threads to stop
        stop = true;

        // Wake up all sleeping threads
        for (auto& queue : queues) {
            std::lock_guard<std::mutex> lock(queue->mtx);
            queue->cv.notify_all();  // Notify all waiting threads
        }
        for (auto& t : threads) {
            if (t.joinable()) t.join();  // Clean up thread resources
        }

        #ifdef DEBUG
            // Print execution time statistics for each thread
            for (size_t i = 0; i < thread_exec_times.size(); ++i) {
                auto nanos = std::chrono::nanoseconds(thread_exec_times[i].load());
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(nanos);
                std::cout << "\033[32m[Thread]\033[0m " << i << " total task execution time: "
                        << millis.count() << " ms" << std::endl;
            }
        #endif
    }

    /**
     * @brief Enqueues a task for execution by the thread pool_.
     *
     * Wraps the callable in a packaged_task to support futures,
     * randomly selects a worker queue for load balancing,
     * and returns a future to track task completion.
     *
     * @tparam FunctionType Callable type (function, lambda, etc.)
     * @tparam ArgumentTypes Argument types for the callable
     * @param task_function Callable to execute
     * @param args Arguments to forward to the callable
     * @return std::future<ReturnType> Future for the task's result
     */
    template <class FunctionType, class... ArgumentTypes>
    auto Enqueue(FunctionType&& task_function, ArgumentTypes&&... args)
            -> std::future<decltype(task_function(args...))>{
        // Get the return type of the task function
        using ReturnType = decltype(task_function(args...));

        // Package the task into a future so we can track its result
        // Using shared_ptr ensures the task remains alive until execution
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<FunctionType>(task_function),
                std::forward<ArgumentTypes>(args)...));

        // Wrap the task so we can decrease the active task count when it's done
        // This lambda captures 'this' and must be thread-safe
        std::function<void()> wrapped = [this, task]() {
            (*task)();  // Run the actual task

            // If this was the last active task, notify anyone waiting
            {
                std::unique_lock<std::mutex> lock(wait_mutex);
                if (--active_tasks == 0) {
                    wait_cv.notify_all();
                }
            }
        };

        // Step 1: Select a worker queue for this task
        size_t index;
        {
            // Lock ensures active_tasks count is updated safely across threads
            std::lock_guard<std::mutex> lock(wait_mutex);
            active_tasks++;  // Increase task counter
            index = dist(rng) % queues.size();  // Pick a queue randomly
        }

        // Step 2: Add task to the selected queue
        {
            // Lock ensures no other thread modifies this queue at the same time
            std::lock_guard<std::mutex> queue_lock(queues[index]->mtx);

            // Adding the task to the front ensures LIFO (Last In, First Out) execution
            queues[index]->tasks.emplace_front(std::move(wrapped));

            // Notify one waiting worker thread that a new task is available
            queues[index]->cv.notify_one();
        }

        // Return a future so the caller can wait for the task result
        return task->get_future();
    }

    /**
     * @brief Blocks until all enqueued tasks have completed.
     *
     * Uses a condition variable to wait for the active task count
     * to reach zero. Useful for synchronization points.
     */
    void WaitAll() {
        std::unique_lock<std::mutex> lock(wait_mutex);
        // Wait until no active tasks remain
        wait_cv.wait(lock, [this]() {
            return active_tasks == 0;  // Predicate to check completion
        });
    }

 private:
    /**
     * @brief Worker thread task queue structure.
     *
     * Contains a deque of tasks, mutex for synchronization,
     * and condition variable for notification.
     */
    struct WorkerQueue {
        std::deque<std::function<void()>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
    };

    /**
     * @brief Main worker thread execution loop.
     *
     * Continuously attempts to get tasks either from its own queue
     * or by stealing from other threads. Tracks execution time
     * for performance monitoring.
     *
     * @param index Worker thread index (0 to thread_count-1)
     */
    void WorkerLoop(size_t index) {
        using clock = std::chrono::steady_clock;

        while (!stop) {
            std::function<void()> task;

            // Try to get work from own queue first
            if (PopTask(index, task)) {
                ExecuteTask(index, task);
            } else if (work_stealing_enabled && StealTask(index, task)) {
                // Then try stealing if enabled and no local work
                ExecuteTask(index, task);
            } else {
                // No work available
                std::unique_lock<std::mutex> lock(queues[index]->mtx);
                queues[index]->cv.wait_for(lock, std::chrono::milliseconds(100));
            }
        }
    }

    /**
     * @brief Attempts to pop a task from the thread's own queue.
     *
     * @param index Worker thread index
     * @param task [out] Retrieved task if successful
     * @return true if a task was retrieved, false otherwise
     */
    bool PopTask(size_t index, std::function<void()>& task) {
        std::lock_guard<std::mutex> lock(queues[index]->mtx);

        // Check if queue has tasks
        if (!queues[index]->tasks.empty()) {
            // Take task from front of the queue (LIFO order)
            task = std::move(queues[index]->tasks.front());
            queues[index]->tasks.pop_front();

            return true;
        }

        return false;
    }

    /**
    * @brief Attempts to steal a task from another thread's queue.
    *
    * Implements work stealing to improve load balancing by taking tasks from
    * the back of other threads' queues (FIFO order) while the owner thread
    * takes tasks from the front (LIFO order), reducing contention.
    *
    * @param thief_index Index of the stealing thread (not used for queue selection)
    * @param[out] task Retrieved task if successful
    * @return true if a task was stolen, false if all queues were empty
    *
    * @warning Concurrency risks:
    * - Potential data race detected by -fsanitize=thread
    * - While mutex-protected, race conditions may still occur in high contention
    * - Risk case: Multiple threads stealing from same queue simultaneously
    * 
    * @note Safety measures:
    * - Uses try_lock to prevent deadlocks
    * - Entire steal operation is atomic under lock
    */
    bool StealTask(size_t thief_index, std::function<void()>& task) {
        // Total number of worker queues (equal to thread count)
        size_t n = queues.size();

        // Circular search through all possible victim queues
        for (size_t i = 0; i < n; ++i) {
            // Calculate victim thread index using modulo arithmetic
            // Starts from thief_index+1 and wraps around
            size_t victim = (thief_index + i + 1) % n;

            // Attempt to acquire lock without blocking
            // If queue is already locked, skip to next candidate
            std::unique_lock<std::mutex> lock(queues[victim]->mtx, std::try_to_lock);
            if (!lock.owns_lock()) continue;

            // Only proceed if queue is not empty
            // Check is done while holding the lock for atomicity
            if (!queues[victim]->tasks.empty()) {
                // Move task from victim's queue to output parameter
                // Using back()+pop_back() instead of front()+pop_front() because:
                // 1. Reduces contention with owner thread that uses front
                // 2. Better for cache locality in many cases
                task = std::move(queues[victim]->tasks.back());
                queues[victim]->tasks.pop_back();
                return true;
            }
        }

        // No tasks found in any queue
        return false;
    }


    /**
    * @brief Executes a task and records its execution time
    * @param index Worker thread index for stats tracking
    * @param task The task function to execute
    */
    void ExecuteTask(size_t index, std::function<void()>& task) {
        auto start = std::chrono::steady_clock::now();  // Start timer
        task();                                                     // Execute task
        auto end = std::chrono::steady_clock::now();    // Stop timer

        // Record duration in thread's execution time total
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        thread_exec_times[index].fetch_add(duration);  // Atomic update
    }

    // Thread control
    std::atomic<bool> stop;                               // Flag to signal thread termination
    std::vector<std::thread> threads;                     // Worker thread collection
    std::vector<std::unique_ptr<WorkerQueue>> queues;     // Per-thread task queues

    // Task distribution
    std::mt19937 rng { std::random_device {}() };     // Random number generator
    std::uniform_int_distribution<size_t> dist;           // Distribution for queue selection
    const bool work_stealing_enabled;                     // Work stealing (Default true)

    // Task tracking
    std::atomic<size_t> active_tasks{0};                  // Count of currently active tasks
    std::mutex wait_mutex;                                // Mutex for task completion waiting
    std::condition_variable wait_cv;                      // CV for task completion notification

    // Performance metrics
    std::vector<std::atomic<int64_t>> thread_exec_times;  // ms exesute time tracking per thread
};  // class ThreadPool

/**
 * @class UnDirectedGraph
 * @brief A thread-safe undirected graph implementation with weighted edges and parallel processing capabilities.
 *
 * Features include:
 * - Weighted edge management
 * - Parallel DFS for connected components
 * - Bidirectional Dijkstra's algorithm for shortest paths
 * - File I/O operations for graph persistence
 * - Optimized data structures for large-scale graphs
 */
template <typename NodeType>
class UnDirectedGraph {
public:
    /**
     * @brief Constructs an empty undirected graph.
     *
     * Initializes the graph with thread count equal to hardware concurrency.
     * Prepares internal data structures with optimized capacity.
     */
    UnDirectedGraph() : pool_(std::thread::hardware_concurrency()) {}

    /**
     * @brief Adds an undirected edge between two nodes with specified weight.
     *
     * Creates nodes if they don't exist and establishes bidirectional connection.
     * Thread-safe through internal ID mapping.
     *
     * @param source Source node identifier
     * @param neighbor Connected node identifier
     * @param weight Edge weight (must be positive)
     */
    void AddEdge(const NodeType& source, const NodeType& neighbor, float weight) {
        size_t src_id = GetOrCreateId(source);
        size_t nb_id = GetOrCreateId(neighbor);
        
        adj_list_[src_id].emplace_back(Edge{nb_id, weight});
        adj_list_[nb_id].emplace_back(Edge{src_id, weight});
    }  // AddEdge()

    /**
     * @brief Saves adjacency list representation to file.
     *
     * Output format includes:
     * - Node listings with connected edges
     * - Weight information for each edge
     * - Formatted columns for readability
     *
     * @param file_number Numeric suffix for output filename (e.g., "1" -> "pairs1.adj")
     */
    void SaveToAdjFile(const std::string& file_number) {
        std::string file_name = "pairs" + file_number + ".adj";
        std::ofstream adj_output(file_name);
        if (!adj_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        SortAdjList();
        std::ostringstream buffer;
        buffer << "<<< There are " << id_to_node_.size() << " IDs in total. >>>\n";

        size_t node_index = 0;
        for (size_t id = 0; id < id_to_node_.size(); ++id) {
            if (adj_list_[id].empty()) continue;

            ++node_index;
            buffer << "[" << std::setw(3) << node_index << "] " 
                   << id_to_node_[id] << ": \n";

            size_t line_counter = 1;
            for (size_t i = 0; i < adj_list_[id].size(); ++i) {
                const auto& edge = adj_list_[id][i];
                buffer << "\t(" << std::setw(2) << i + 1 << ") "
                       << id_to_node_[edge.neighbor_id] << "," 
                       << std::setw(7) << edge.weight;

                if (line_counter++ % 12 == 0) buffer << '\n';
            }
            buffer << '\n';
        }

        buffer << "<<< There are " << node_count << " nodes in adjacency lists. >>>\n";
        adj_output << buffer.str();
    }  // SaveToAdjFile()

    /**
     * @brief Saves connected components analysis to file.
     *
     * Components are sorted by:
     * 1. Size (descending)
     * 2. First node ID (ascending)
     *
     * @param file_number Numeric suffix for output filename (e.g., "1" -> "pairs1.cc")
     */
    void SaveToCCFile(const std::string& file_number) {
        std::string file_name = "pairs" + file_number + ".cc";
        std::ofstream cc_output(file_name);
        if (!cc_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        // Sort connected components 
        // (in descending order by size, 
        //  and in ascending order by the first element if the sizes are the same)
        std::vector<size_t> sorted_indices(component_sets_.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [this](size_t i, size_t j) {
                if (component_sets_[i].size() != component_sets_[j].size()) {
                    return component_sets_[i].size() > component_sets_[j].size();
                }
                return component_sets_[i].front() < component_sets_[j].front();
            });

        std::ostringstream buffer_cout;
        std::ostringstream buffer_file;
        
        std::string msg = "<<< There are " + std::to_string(sorted_indices.size()) 
                        + " connected components in total. >>>\n";
        buffer_cout << msg;
        buffer_file << msg;

        size_t counter = 0;
        for (const size_t& index : sorted_indices) {
            if (component_sets_[index].empty()) continue;

            size_t set_size = component_sets_[index].size();
            ++counter;
            
            buffer_cout << "{" << std::setw(2) << counter 
                    << "} Connected Component: size = " << set_size << "\n";
            
            buffer_file << "{" << std::setw(2) << counter 
                    << "} Connected Component: size = " << set_size << "\n";

            size_t node_counter = 0;
            for (size_t node_id : component_sets_[index]) {
                ++node_counter;
                if ((node_counter % 8) == 1) {  
                    buffer_file << " \t";
                }
                
                buffer_file << "(" << std::setw(3) << node_counter << ") " 
                        << id_to_node_[node_id];

                if ((node_counter % 8) == 0 || node_counter == set_size) {  
                    buffer_file << '\n';
                } else {
                    buffer_file << "\t";
                }
            }
        }

        std::cout << buffer_cout.str();
        cc_output << buffer_file.str();
    }  // SaveToCCFile()

    /**
     * @brief Saves shortest path results to file.
     *
     * Output includes:
     * - Source node identifier
     * - Sorted list of destinations with distances
     * - Formatted tabular output
     *
     * @param file_number Numeric suffix for output filename
     * @param source Origin node identifier for path results
     */
    void SaveToDsFile(const std::string& file_number, const NodeType& source) {
        #ifdef DEBUG
                auto start_time3 = std::chrono::high_resolution_clock::now();
        #endif
        std::string file_name = "pairs" + file_number + ".ds";
        std::ofstream ds_output(file_name);
        if (!ds_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        std::ostringstream buffer;
        buffer << "\norigin: " << source << "\n";

        size_t result_count = 0;
        for (const auto& result : path_min_value_) {
            if (result.node_id == node_to_id_[source]) continue;

            buffer << "(" << std::setw(2) << ++result_count << ") \t"
                << id_to_node_[result.node_id] << ", " << result.weight;

            if (result_count % 8 == 0 || result_count == path_min_value_.size()) {
                buffer << "\t\n";
            } else {
                buffer << "\t";
            }
        }

        buffer << "\n";
        ds_output << buffer.str();
        path_min_value_.clear();

        #ifdef DEBUG
            std::mutex log_mtx_;
            auto end_time3 = std::chrono::high_resolution_clock::now();
            auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time3 - start_time3);

            DEBUG_LOG("\033[33m Ds Execution Time: " << duration3.count() << " ns\033[0m\n");
        #endif
    }  // SaveToDsFile()

    /**
     * @brief Executes parallel DFS to identify all connected components.
     *
     * Features:
     * - Depth-limited work splitting for parallelization
     * - Atomic component tracking
     * - Optimized visited node marking
     */
    void RunDFS() {
        #ifdef DEBUG
            auto start_time = std::chrono::high_resolution_clock::now();
        #endif
        visited_.assign(id_to_node_.size(), false);
        next_component_ = 0;
        component_sets_.clear();
        component_id_.assign(id_to_node_.size(), -1);

        for (size_t id = 0; id < id_to_node_.size(); ++id) {
            if (!visited_[id]) {
                TryStartDFS(id);
            }
        }

        visited_.clear();
        #ifdef DEBUG
            std::mutex log_mtx_;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

            DEBUG_LOG("\033[33m DFS Execution Time: " << duration.count() << " ns\033[0m\n");
        #endif
    }  // RunDFS()

    /**
    * @brief Computes shortest paths from a source node to all other nodes in its component using Dijkstra's algorithm.
    * 
    * This implementation:
    * 1. Performs a single Dijkstra pass from the source node
    * 2. Filters results to only include nodes in the same connected component
    * 3. Returns sorted results by ascending distance
    * 
    * @param source The starting node identifier
    * @return true if computation succeeded, false if source node doesn't exist
    */
    bool ComputeShortestDistance(const NodeType& source) {
        #ifdef DEBUG
            auto start_time = std::chrono::high_resolution_clock::now();
        #endif

        path_min_value_.clear();

        // Locate source node ID in the mapping
        auto src_it = node_to_id_.find(source);
        if (src_it == node_to_id_.end()) {
            std::cerr << "Source node not found.\n";
            return false;
        }
        size_t src_id = src_it->second;

        // Retrieve the connected component ID for the source node
        size_t cid = component_id_[src_id];

        // Compute shortest paths to all reachable nodes
        std::vector<float> dist_map = DijkstraAll(src_id);  // Use vector instead of unordered_map

        // Collect distances for nodes in the same component (excluding source)
        for (size_t target_id : component_sets_[cid]) {
            if (target_id == src_id) continue;

            // Use stored distance if available, otherwise mark as unreachable (INF)
            float distance = (target_id < dist_map.size()) ? dist_map[target_id] : INF;
            path_min_value_.emplace_back(Result{target_id, distance});
        }

        // Sort results primarily by distance, secondarily by node ID
        std::sort(path_min_value_.begin(), path_min_value_.end(), 
            [](const auto& a, const auto& b) {
                return std::tie(a.weight, a.node_id) < std::tie(b.weight, b.node_id);
            });

        #ifdef DEBUG
            std::mutex log_mtx_;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            DEBUG_LOG("\033[33mSource: " << src_id << " | DijkstraAll execution time: " 
                                         << duration.count() << " ns\033[0m\n");
        #endif

        return true;
    }

    /**
     * @brief Prints all node identifiers in sorted order.
     *
     * Output format:
     * - 8 nodes per line
     * - Right-aligned columns
     * - Sorted lexicographically
     */
    void PrintKeyList() {
        std::ostringstream buffer;
        size_t node_counter = 0;
        
        // Sort by node name and output
        std::vector<size_t> sorted_ids(id_to_node_.size());
        std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
        std::sort(sorted_ids.begin(), sorted_ids.end(),
            [this](size_t a, size_t b) {
                return id_to_node_[a] < id_to_node_[b];
            });

        for (size_t id : sorted_ids) {
            buffer << std::right << std::setw(12) << id_to_node_[id];
            if (++node_counter % 8 == 0) {
                buffer << '\n';
            }
        }

        if (node_counter % 8 != 0) {
            buffer << '\n';
        }

        std::cout << buffer.str();
    }  // PrintKeyList()

    bool Empty() const { return id_to_node_.empty(); }
    size_t GetNodeCount() const { return node_count; }
    size_t GetUniqueNodeCount() const { return id_to_node_.size(); }

    void Clear() {
        adj_list_.clear();
        node_to_id_.clear();
        id_to_node_.clear();
        component_sets_.clear();
        component_id_.clear();
        visited_.clear();
        path_min_value_.clear();
        node_count = 0;
        next_component_ = 0;
    }

    void Perf() {
        node_to_id_.max_load_factor(0.25);
        node_to_id_.reserve(8 * 1024);
        id_to_node_.reserve(8 * 1024);
        adj_list_.reserve(8 * 1024);
        component_sets_.reserve(8 * 1024);
        component_id_.reserve(8 * 1024);
        visited_.reserve(8 * 1024);
        path_min_value_.reserve(8 * 1024);
    }

 private:
    /**
     * @struct Edge
     * @brief Represents a weighted connection between nodes.
     */
    struct Edge {
        size_t neighbor_id;
        float weight;
    };

    /**
     * @struct Result
     * @brief Stores shortest path computation results.
     */
    struct Result {
        size_t node_id;
        float weight;
    };

    /**
     * @brief Retrieves or creates unique integer ID for a node.
     *
     * Maintains bidirectional mapping between NodeType and size_t.
     * Thread-safe through atomic operations on internal structures.
     *
     * @param node Graph node identifier
     * @return Unique integer ID for the node
     */
    size_t GetOrCreateId(const NodeType& node) {
        auto it = node_to_id_.find(node);
        if (it != node_to_id_.end()) return it->second;
        
        size_t new_id = id_to_node_.size();
        node_to_id_[node] = new_id;
        id_to_node_.push_back(node);
        adj_list_.emplace_back();
        return new_id;
    }

    /**
     * @brief Sorts adjacency lists by node identifier.
     *
     * Parallel processing features:
     * - Batched node processing
     * - Concurrent map updates
     * - Lock-free operations where possible
     */
    void SortAdjList() {
        #ifdef DEBUG
            auto start_time = std::chrono::high_resolution_clock::now();
        #endif

        // Sort the node IDs by name (this part is difficult to parallelize, keep single thread)
        std::vector<size_t> sorted_ids(id_to_node_.size());
        std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
        std::sort(sorted_ids.begin(), sorted_ids.end(),
            [this](size_t a, size_t b) {
                return id_to_node_[a] < id_to_node_[b];
            });

        // Remap ID (single-threaded fast operation)
        std::vector<size_t> new_ids(id_to_node_.size());
        for (size_t i = 0; i < sorted_ids.size(); ++i) {
            new_ids[sorted_ids[i]] = i;
        }

        // Prepare new data structure
        std::vector<NodeType> new_id_to_node(id_to_node_.size());
        std::unordered_map<NodeType, size_t> new_node_to_id;
        std::vector<std::vector<Edge>> new_adj_list(id_to_node_.size());
        
        // Rebuild data structure in parallel
        std::mutex map_mutex;
        std::vector<std::future<void>> futures;
        
        // Batch processing nodes
        const size_t batch_size = 50;
        for (size_t start = 0; start < sorted_ids.size(); start += batch_size) {
            futures.emplace_back(pool_.Enqueue([&, start, batch_size]() {
                size_t end = std::min(start + batch_size, sorted_ids.size());
                std::vector<std::pair<NodeType, size_t>> local_entries;
                local_entries.reserve(end - start);
                
                for (size_t i = start; i < end; ++i) {
                    size_t old_id = sorted_ids[i];
                    new_id_to_node[i] = id_to_node_[old_id];
                    local_entries.emplace_back(new_id_to_node[i], i);
                    
                    // Process the adjacency list
                    auto& new_edges = new_adj_list[i];
                    new_edges.reserve(adj_list_[old_id].size());
                    for (const Edge& edge : adj_list_[old_id]) {
                        new_edges.emplace_back(Edge{new_ids[edge.neighbor_id], edge.weight});
                    }
                    
                    // Sort the adjacency list of the current node
                    std::sort(new_edges.begin(), new_edges.end(),
                        [](const Edge& a, const Edge& b) {
                            return a.neighbor_id < b.neighbor_id;
                        });
                }
                
                // Batch update mapping table
                std::lock_guard<std::mutex> lock(map_mutex);
                for (auto& entry : local_entries) {
                    new_node_to_id.insert(entry);
                }
            }));
        }
        
        // Wait
        for (auto& f : futures) f.get();

        // Move data to member variables
        id_to_node_ = std::move(new_id_to_node);
        node_to_id_ = std::move(new_node_to_id);
        adj_list_ = std::move(new_adj_list);
        
        // Calculate the total number of edges
        node_count = 0;
        for (const auto& edges : adj_list_) {
            node_count += edges.size();
        }

        #ifdef DEBUG
            std::mutex log_mtx_;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            DEBUG_LOG("\033[33mParallel SortAdjList executed in " << duration.count() << " ms\033[0m\n");
        #endif
    }

    /**
     * @brief Initiates new connected component tracking.
     *
     * Atomically reserves component ID and begins DFS traversal.
     *
     * @param start_id Node identifier to begin component discovery
     */
    void TryStartDFS(size_t start_id) {
        size_t cid = next_component_++;
        component_sets_.emplace_back();  // Add an empty vector
        InternalDFS(start_id, cid);
    }

    /**
     * @brief Internal DFS implementation with parallel work splitting.
     *
     * Features:
     * - Iterative implementation (avoids stack overflow)
     * - Depth-based work delegation to thread pool
     * - Atomic visited marking
     *
     * @param node_id Current node identifier
     * @param cid Connected component identifier
     */
    void InternalDFS(size_t node_id, size_t cid) {
        std::deque<std::pair<size_t, size_t>> stack;
        stack.emplace_back(node_id, 0);

        while (!stack.empty()) {
            auto [current_id, depth] = stack.back();
            stack.pop_back();

            if (visited_[current_id]) continue;
            visited_[current_id] = true;
            component_id_[current_id] = cid;
            
            // Use ordered insertion to ensure uniqueness
            auto& comp = component_sets_[cid];
            auto it = std::lower_bound(comp.begin(), comp.end(), current_id);
            if (it == comp.end() || *it != current_id) {
                comp.insert(it, current_id);
            }

            if (depth >= DEPTH_CUTOFF) {
                pool_.Enqueue([this, current_id, cid]() {
                    InternalDFS(current_id, cid);
                });
                continue;
            }

            for (const Edge& edge : adj_list_[current_id]) {
                if (!visited_[edge.neighbor_id]) {
                    stack.emplace_back(edge.neighbor_id, depth + 1);
                }
            }
        }
    }  // InternalDFS()

    /**
    * @brief Executes Dijkstra's algorithm to find shortest paths from a source node to all other nodes.
    * 
    * Implementation features:
    * - Uses min-heap for efficient extraction of the next closest node
    * - Tracks visited nodes to avoid reprocessing
    * - Stores distances in a hash map for dynamic growth
    * - Processes all reachable nodes from the source
    * 
    * @param src_id The source node ID
    * @return unordered_map<size_t, float> Mapping of node IDs to their shortest path distances
    */
    std::vector<float> DijkstraAll(size_t src_id) {
        const size_t N = id_to_node_.size();  // Total number of nodes
        std::vector<float> dist(N, INF);      // Distance vector, initialized to infinity
        std::vector<bool> visited(N, false);  // Tracks processed nodes

        // Min-heap priority queue: <distance, node_id>
        using MinHeap = std::priority_queue<std::pair<float, size_t>,
                                            std::vector<std::pair<float, size_t>>,
                                            std::greater<>>;
        MinHeap pq;

        // Initialize with the source node
        dist[src_id] = 0.0f;
        pq.emplace(0.0f, src_id);

        while (!pq.empty()) {
            // Extract the node with the shortest known distance
            auto [dist_u, u] = pq.top();
            pq.pop();

            // Skip if already processed, preventing duplicate calculations
            if (visited[u]) continue;
            visited[u] = true;

            // Relax all outgoing edges
            for (const Edge& edge : adj_list_[u]) {
                size_t v = edge.neighbor_id;
                float new_dist = dist[u] + edge.weight;

                // Update distance if a shorter path is found
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pq.emplace(new_dist, v);  // Push into the priority queue for further exploration
                }
            }
        }

        return dist;  // Returns shortest distances from the source to all nodes
    }  // DijkstraAll()

    // Member variables
    std::unordered_map<NodeType, size_t> node_to_id_;
    std::vector<NodeType> id_to_node_;
    std::vector<std::vector<Edge>> adj_list_;
    size_t node_count = 0;

    // Connected components
    std::vector<std::vector<size_t>> component_sets_;
    std::vector<size_t> component_id_;
    std::atomic<size_t> next_component_{0};

    // Traversal status
    std::vector<bool> visited_;
    const size_t DEPTH_CUTOFF = 128;

    // Shortest path result
    std::vector<Result> path_min_value_;

    // Thread pool
    ThreadPool pool_;

    // Constants
    const float INF = std::numeric_limits<float>::infinity();
};

static bool IsFloat(const std::string& str) {
    std::regex floatRegex(R"(^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$)");
    return std::regex_match(str, floatRegex);
}  // IsFloat()

static bool IsNumeric(const std::string& str) {
    if (str.empty()) return false;

    size_t i = 0;
    bool hasDecimal = false;

    // Negative sign is not allowed
    if (str[i] == '-') return false;

    // Leading '.' is allowed
    if (str[i] == '.') {
        hasDecimal = true;
        ++i;
    }

    for (; i < str.size(); ++i) {
        if (str[i] == '.') {
            if (hasDecimal) return false; // There can only be one decimal point

            hasDecimal = true;
        } else if (!isdigit(str[i])) {
            return false; // Not a number
        }
    }

    return i > 0; // Make sure there is at least one valid digit
}  // IsNumeric()

/**
 * @brief Reads student records from a binary file and constructs a directed graph
 * 
 * @tparam T Data type (must be StudentType)
 * @param file_number File number suffix (e.g., "1" for "pairs1.bin")
 * @param graph [out] Perf instance to be populated with the data
 */
template <typename T>
static void ReadBinary(const std::string& file_number, UnDirectedGraph<T>& graph, const float& value) {
    std::string file_name = "pairs" + file_number + ".bin";

    // Open file in binary mode and determine size
    std::ifstream file(file_name, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open binary file: " << file_name << "\n";
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Verify file size alignment with StudentType
    if (size % sizeof(StudentType) != 0) {
        std::cerr << "Binary file size is not aligned with StudentType size.\n";
        return;
    }

    // Calculate number of records
    size_t count = size / sizeof(StudentType);
    std::vector<StudentType> buffer(count);

    // Read binary data into buffer
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read binary data.\n";
        return;
    }

    file.close();

    // Populate graph with edges from the records
    for (const auto& student : buffer) {
        if ((student.weight <= value && student.weight > 0) && student.source != student.neighbor) {
            graph.AddEdge(student.source, student.neighbor, student.weight);
        }
    }
}  // ReadBinary()

/**
 * @brief Task 1: Build adjacency lists from binary data and save results
 * 
 * 1. Reads student records from binary file
 * 2. Constructs directed graph adjacency lists
 * 3. Saves formatted results to output file
 * 
 * @param undir_graph Perf instance to operate on
 * @param file_number File number suffix for input/output files
 */
static void Task1(UnDirectedGraph<std::string>& undir_graph, std::string& file_number, std::string& file_name) {
    std::string input_float = "";
    float value = 0.0f;
    // User input validation loop
    do {
        std::cout << "Input a real number in (0,1]: " << std::flush;
        std::cin >> input_float;
        std::string err_msg = "### " + input_float + " is NOT in (0,1] ###\n";

        if (!IsNumeric(input_float)) {
            std::cout << '\n';
            continue;
        }

        // Validate float input
        if (IsFloat(input_float)) {
            value = std::stof(input_float);

            input_float.erase(input_float.find_last_not_of('0') + 1);

            // Validate range
            if (0.0f < value && value <= 1.0f) {
                break;
            } else {
                std::cout << '\n';
                std::cout << err_msg;
            }
        } else {
            std::cout << '\n';
            std::cout << err_msg;
        }

        std::cout << '\n';
    } while (true);

    std::cout << '\n';

    std::cout << "Input a file number ([0] Quit): " << std::flush;
    std::cin >> file_number;

    if (file_number == "0") {
        return;
    } else {
        // `struct stat` stores file metadata like size, permissions, and timestamps.
        struct stat buffer;
        file_name = "pairs" + file_number + ".bin";
        std::cout << '\n';

        // Check file existence
        if (stat(file_name.c_str(), &buffer)) {
            std::cout << '\n';
            std::cout << "### " << file_name << " does not exist! ###\n";

            return;
        }
    }

    ReadBinary(file_number, undir_graph, value);

    if (!undir_graph.Empty()) {
        file_number = file_number + "_" + input_float;
        undir_graph.SaveToAdjFile(file_number);

        std::cout << "<<< There are " << undir_graph.GetUniqueNodeCount() << " IDs in total. >>>\n\n"
                  << "<<< There are " << undir_graph.GetNodeCount() << " nodes in adjacency lists. >>>\n\n";

        #ifdef DEBUG
            auto start_time = std::chrono::high_resolution_clock::now();
        #endif

        undir_graph.RunDFS();
        undir_graph.SaveToCCFile(file_number);

        #ifdef DEBUG
            std::mutex log_mtx_;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

            DEBUG_LOG("\033[33m Task1 Execution Time: " << duration.count() << " ns\033[0m\n");
        #endif
    }

}  // Task1()

/**
 * @brief Task 1: Build graph from binary data and analyze connections
 * 
 * 1. Prompts user for weight threshold (0,1]
 * 2. Reads student records from specified binary file
 * 3. Constructs undirected graph with edges meeting weight threshold
 * 4. Saves adjacency lists to .adj file
 * 5. Finds and saves connected components to .cc file
 * 
 * @param undir_graph Graph instance to operate on
 * @param file_number[out] File number suffix for input/output files (gets modified)
 * @param file_name[out] Full input filename (gets set)
 */
static void Task2(UnDirectedGraph<std::string>& undir_graph, const std::string& file_number) {
    std::string source = "";

    do {
        undir_graph.PrintKeyList();

        std::cout << "Input a student ID [0: exit] " << std::flush;
        std::cin >> source;

        if (source == "0") return;

        #ifdef DEBUG
            auto start_time = std::chrono::high_resolution_clock::now();
        #endif

        bool is_exist = undir_graph.ComputeShortestDistance(source);

        std::cout << '\n';
        if (!is_exist) {
            std::cout << "### the student id does not exist! ###\n";
        } else {
            undir_graph.SaveToDsFile(file_number, source);
        }

        #ifdef DEBUG
            std::mutex log_mtx_;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

            DEBUG_LOG("\033[33m Task2 Execution Time: " << duration.count() << " ns\033[0m\n");
        #endif

        std::cout << '\n';
    } while (true);
}  // Task2()

/**
 * @brief Main program entry point with menu-driven interface
 * 
 * Provides interactive menu for graph operations:
 * 0. Quit program
 * 1. Build a graph and connected components (Task 1)
 * 2. Find shortest paths by Dijkstra (Task 2)
 * 
 * Handles input validation and coordinates task execution
 * 
 * @return int Program exit status (0 for normal termination)
 */
int main() {
    FAST_IO();  // Enable fast I/O operations

    int select_command = 0;
    std::string file_number;
    std::string file_name;
    UnDirectedGraph<std::string> undir_graph;



    undir_graph.Perf();  // Initialize graph with optimal settings

    do {
        while (true) {
            // Display the menu options for the user
            std::cout <<
                "**********  Graph data applications  *********\n"
                "* 1. Build a graph and connected components  *\n"
                "* 2. Find shortest paths by Dijkstra         *\n"
                "**********************************************\n"
                "Input a choice(0, 1, 2) [0: QUIT]: "
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
                std::cout << "The command does not exist!\n";
                std::cout << '\n';
            }
        }

        // Handle the different options based on the user's choice
        switch (select_command) {
        case 0:
            break;
        case 1:
            if (!undir_graph.Empty()) {
                undir_graph.Clear();  // Reset existing graph
                undir_graph.Perf();
            }

            std::cout << '\n';
            Task1(undir_graph, file_number, file_name);
            if (undir_graph.Empty()) {
                std::cout << "### There is no graph and try it again. ###\n";
            }
            std::cout << '\n';

            break;
        case 2:
            if (undir_graph.Empty()) {
                std::cout << "### There is no graph and choose 1 first. ###\n";
            } else {
                // Execute Task 2
                std::cout << '\n';
                Task2(undir_graph, file_number);
            }
            std::cout << '\n';

            break;
        default:
            std::cout << '\n';
            std::cout << "The command does not exist!\n";
            std::cout << '\n';
        }
    } while (select_command != 0);  // Continue until the user selects option 0 (quit)

    return 0;
}  // main()