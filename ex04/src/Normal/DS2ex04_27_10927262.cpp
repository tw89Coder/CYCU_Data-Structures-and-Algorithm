/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex04_27_10927262.cpp
 * @brief A program that implements a directed graph with weighted edges and parallel processing capabilities.
 * @version 1.5.1
 *
 * @details
 * This program implements a directed graph data structure that supports:
 * - Adding weighted edges between nodes
 * - Computing reachability between nodes
 * - Parallel processing of graph algorithms
 * - Saving graph data to files in specific formats
 * - Performance-optimized operations for large graphs
 *
 * The user can perform operations such as building adjacency lists, computing connection counts,
 * and analyzing graph connectivity.
 *
 * @author 
 * - Group 27
 * - 10927262 呂易鴻
 */

// Other system headers
#include <sys/stat.h>          // File system status (POSIX)

// C++ Standard Library Headers (alphabetical order)
#include <algorithm>           // Algorithms (sorting, searching, transforming)
#include <atomic>              // Atomic operations and thread synchronization
#include <chrono>              // Time utilities (clocks, time points)
#include <condition_variable>  // Thread synchronization (wait/notify)
#include <deque>               // Double-ended queue container
#include <fstream>             // File stream operations (input/output file streams)
#include <functional>          // Function objects and type erasure (std::function)
#include <iomanip>             // I/O formatting (setw, setprecision)
#include <iostream>            // Standard I/O streams (cin, cout, cerr)
#include <limits>              // Numeric limits (numeric_limits)
#include <memory>              // Smart pointers and dynamic memory management
#include <mutex>               // Mutual exclusion (mutexes, lock guards)
#include <future>              // Future/promise for asynchronous operations
#include <random>              // Random number generation
#include <queue>               // Queue container (FIFO)
#include <sstream>             // String streams (in-memory I/O)
#include <string>              // String class and operations
#include <thread>              // Thread support (std::thread)
#include <utility>             // Utility components (pair, move, forward)
#include <unordered_set>       // Unordered associative container (hash set)
#include <unordered_map>       // Unordered associative container (hash map)
#include <vector>              // Dynamic array container

// C Standard Library (C++ wrapper headers, alphabetical order)
#include <cstddef>             // Fundamental types (size_t, nullptr_t)
#include <cstdlib>             // General utilities (malloc, exit, atoi)
#include <cstring>             // C-style string operations (strcpy, memcmp)

// Constant definition
#define MAX_LEN 12    // Array size of student id and name.

// Input/Output optimization
#define FAST_IO() \
    /* Disables synchronization with C-style I/O for faster execution */ \
    std::ios_base::sync_with_stdio(false); \
    /* Unbinds cin from cout to avoid unnecessary flushing */ \
    std::cin.tie(nullptr); \
     /* Unbinds cout to improve independent output performance */ \
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
    char publisher[MAX_LEN];
    char subscriber[MAX_LEN];
    float weight;
} StudentType;

/**
 * @brief A thread pool implementation with work stealing capabilities.
 *
 * Manages a pool of worker threads that execute enqueued tasks efficiently.
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
    explicit ThreadPool(size_t thread_count)
        : stop(false), threads(thread_count), thread_exec_times(thread_count) {
        // Create a dedicated task queue for each worker thread
        for (size_t i = 0; i < thread_count; ++i) {
            queues.emplace_back(std::make_unique<WorkerQueue>());
        }

        // Launch worker threads with their assigned index
        for (size_t i = 0; i < thread_count; ++i) {
            threads[i] = std::thread([this, i] {
                WorkerLoop(i);  // Each thread runs the main work loop
            });
        }
    }

    /**
     * @brief Destructor that safely shuts down the thread pool.
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
                std::cout << "Thread " << i << " total task execution time: "
                        << millis.count() << " ms" << std::endl;
            }
        #endif
    }

    /**
     * @brief Enqueues a task for execution by the thread pool.
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
                 -> std::future<decltype(task_function(args...))> {
        // Determine the return type of the task function
        using ReturnType = decltype(task_function(args...));

        // Create a packaged task to get future and handle exceptions
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<FunctionType>(task_function),
                std::forward<ArgumentTypes>(args)...));

        // Wrap the task to decrement active task count upon completion
        std::function<void()> wrapped = [this, task]() {
            (*task)();  // Execute the actual task

            // Notify if this was the last active task
            if (--active_tasks == 0) {
                std::unique_lock<std::mutex> lock(wait_mutex);
                wait_cv.notify_all();  // Wake up anyone waiting on WaitAll()
            }
        };

        active_tasks++;  // Increment active task counter

        // Randomly select a worker queue for load balancing
        size_t index = dist(rng) % queues.size();

        // Add task to the selected worker's queue
        {
            std::lock_guard<std::mutex> lock(queues[index]->mtx);
            queues[index]->tasks.emplace_front(std::move(wrapped));  // Add to front for LIFO
        }

        // Notify one worker thread in case it's sleeping
        queues[index]->cv.notify_one();

        // Return future for the task result
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

        // Main work loop runs until stop is signaled
        while (!stop) {
            std::function<void()> task;

            // Try to get work from own queue or steal from others
            if (PopTask(index, task) || StealTask(index, task)) {
                // Measure task execution time
                auto start = clock::now();
                task();  // Execute the task
                auto end = clock::now();

                // Record execution time in nanoseconds
                auto duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                thread_exec_times[index].fetch_add(duration);
            } else {
                // No work available - sleep briefly
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
     * Implements work stealing to improve load balancing.
     * Steals from the back of other threads' queues to reduce contention.
     *
     * @param thief_index Index of the stealing thread
     * @param task [out] Retrieved task if successful
     * @return true if a task was stolen, false otherwise
     */
    bool StealTask(size_t thief_index, std::function<void()>& task) {
        size_t n = queues.size();

        // Try stealing from each other thread in round-robin fashion
        for (size_t i = 0; i < n; ++i) {
            size_t victim = (thief_index + i + 1) % n;  // Calculate victim index

            std::lock_guard<std::mutex> lock(queues[victim]->mtx);
            if (!queues[victim]->tasks.empty()) {
                // Take task from back of victim's queue (FIFO order for stealing)
                task = std::move(queues[victim]->tasks.back());
                queues[victim]->tasks.pop_back();

                return true;
            }
        }

        return false;
    }

    // Thread control
    std::atomic<bool> stop;                               // Flag to signal thread termination
    std::vector<std::thread> threads;                     // Worker thread collection
    std::vector<std::unique_ptr<WorkerQueue>> queues;     // Per-thread task queues

    // Task distribution
    std::mt19937 rng { std::random_device{}() };      // Random number generator
    std::uniform_int_distribution<size_t> dist;           // Distribution for queue selection

    // Task tracking
    std::atomic<size_t> active_tasks{0};                  // Count of currently active tasks
    std::mutex wait_mutex;                                // Mutex for task completion waiting
    std::condition_variable wait_cv;                      // CV for task completion notification

    // Performance metrics
    std::vector<std::atomic<int64_t>> thread_exec_times;  // ms exesute time tracking per thread
};  // ThreadPool

/**
 * @brief A directed graph implementation with weighted edges and parallel processing capabilities.
 *
 * This class represents a directed graph where nodes can be of any comparable type (NodeType).
 * It provides functionality for:
 * - Adding directed edges with weights
 * - Computing reachability between nodes
 * - Parallel processing of graph algorithms
 * - Saving graph data to files in specific formats
 * - Performance-optimized operations for large graphs
 */
template <typename NodeType>
class DirectedGraph {
 public:
    /**
     * @brief Adds a directed edge from publisher to subscriber with a given weight.
     * 
     * Maintains the adjacency list and updates node counts. Ensures both nodes exist
     * in the graph even if they only appear as endpoints.
     * 
     * @param publisher The source node of the edge
     * @param subscriber The target node of the edge
     * @param weight The weight associated with the edge
     */
    void AddEdge(const NodeType& publisher, const NodeType& subscriber, float weight) {
        // Ensure publisher exists.
        adj_list[publisher].emplace_back(subscriber, weight);
        ++node_count;

        // If publisher is new, update publisher_list.
        if (adj_list[publisher].size() == 1) {
            ++publisher_count;
            publisher_list.emplace_back(publisher, 0);
        }

        // Ensure subscriber exists as a key in adj_list (even with empty edges).
        if (adj_list.find(subscriber) == adj_list.end()) {
            adj_list[subscriber] = {};  // Insert empty vector
        }
    }

    /**
     * @brief Saves the graph adjacency information to a formatted .adj file.
     * 
     * Output includes:
     * - Sorted publisher nodes
     * - Their connections with weights
     * - Formatted for human readability
     * 
     * @param file_number Suffix for the output filename (e.g., "1" creates "pairs1.adj")
     */
    void SaveToAdjFile(const std::string& file_number) const {
        // Construct output filename
        std::string file_name = "pairs" + file_number + ".adj";
        std::ofstream adj_output(file_name.c_str());

        if (!adj_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        size_t publisher_index = 0;
        SortKeys();  // Sort publisher nodes alphabetically

        std::ostringstream buffer;
        buffer << "<<< There are " << publisher_count << " IDs in total. >>>\n";

        // Process each publisher node
        for (const auto& pair : publisher_list) {
            const NodeType& publisher = pair.first;
            auto graph_node = adj_list.find(publisher);

            // Skip if publisher has no edges
            if (graph_node == adj_list.end() || graph_node->second.empty()) continue;

            std::vector<std::pair<NodeType, float>>& edges = graph_node->second;

            // Choose sorting algorithm based on edge count for performance
            if (edges.size() < 32) {
                // Stable sort preserves order of equal elements for small datasets
                // Insertion sort may be quick for small datasets
                std::stable_sort(edges.begin(), edges.end(),
                    [](const auto& first, const auto& second) {
                        return first.first < second.first;  // Sort by node name
                    });
            } else {
                // Regular sort is faster for larger datasets
                std::sort(edges.begin(), edges.end(),
                    [](const auto& first, const auto& second) {
                        return first.first < second.first;
                    });
            }

            // Format publisher header
            ++publisher_index;
            buffer << "[" << std::setw(3) << publisher_index << "] " << publisher << ": \n";

            // Format output with 12 items per line.
            size_t line_counter = 1;
            for (size_t i = 0; i < edges.size(); ++i) {
                const auto& edge = edges[i];
                buffer << "\t(" << std::setw(2) << i + 1 << ") "
                       << edge.first << "," << std::setw(7) << edge.second;

                // New line every 12 items
                if (line_counter++ == 12) {
                    buffer << '\n';
                    line_counter = 1;
                }
            }

            buffer << '\n';
        }

        // Write footer and flush to file
        buffer << "<<< There are " << node_count << " nodes in total. >>>\n";
        adj_output << buffer.str();
    }

    /**
     * @brief Saves reachability information to a formatted .cnt file.
     * 
     * Output includes:
     * - Sorted publisher nodes with their connection counts
     * - All reachable paths from each publisher
     * - Formatted for human readability
     * 
     * @param file_number Suffix for the output filename (e.g., "1" creates "pairs1.cnt")
     */
    void SaveToCntFile(const std::string& file_number) const {
        std::string file_name = "pairs" + file_number + ".cnt";
        std::ofstream cnt_output(file_name.c_str());

        if (!cnt_output) {
            std::cerr << "Failed to open file: " << file_name << "\n";
            return;
        }

        std::ostringstream buffer;
        buffer << "<<< There are " << publisher_list.size() << " IDs in total. >>>\n";

        // Process each publisher node
        for (size_t i = 0; i < publisher_list.size(); ++i) {
            const NodeType& key = publisher_list[i].first;

            auto graph_node = reachable_vec.find(key);
            if (graph_node != reachable_vec.end()) {
                std::vector<std::vector<NodeType>>& all_paths = graph_node->second;

                // Sort each path for consistent output
                for (auto& path : all_paths) {
                    std::stable_sort(path.begin(), path.end(),
                        [](const NodeType& first, const NodeType& second) {
                            return first < second;  // Sort nodes in path
                        });
                }

                // Format publisher header with connection count
                buffer << "[" << std::setw(3) << i + 1 << "] " << key
                       << "(" << publisher_list[i].second << "): \n";

                // Format output with 12 items per line
                size_t path_idx = 1;
                for (const auto& path : all_paths) {
                    size_t line_counter = 1;

                    // Format nodes in path
                    for (size_t j = 0; j < path.size(); ++j) {
                        buffer << "\t(" << std::setw(2) << j + 1 << ") " << path[j];

                        // New line every 12 items
                        if (line_counter++ == 12) {
                            buffer << '\n';
                            line_counter = 1;
                        }
                    }

                    buffer << '\n';  // End of path
                }
            }
        }

        cnt_output << buffer.str();  // Write all output
    }

    /**
     * @brief Computes connection counts for all nodes using parallel BFS.
     * 
     * Uses a thread pool to parallelize BFS operations across all publisher nodes.
     * Results are stored in reachable_vec and publisher_list.
     * 
     * @param mode Reserved for future use (currently unused)
     * @param min_batch_size Minimum number of nodes to process per thread (default: 4)
     */
    void ComputeAllConnectionCounts(const std::string& mode, size_t min_batch_size = 4) {
        #ifdef DEBUG
            auto start_time = std::chrono::steady_clock::now();
            DEBUG_LOG("Starting ComputeAllConnectionCounts with "
                    << std::thread::hardware_concurrency()
                    << " threads, min_batch_size=" << min_batch_size);
        #endif

        // Local storage for parallel processing
        std::unordered_map<NodeType, std::vector<NodeType>> local_reachables;
        std::unordered_map<NodeType, size_t> local_counts;
        std::mutex result_mutex;  // Protects access to local storage

        {
            // Create thread pool with one thread per CPU core
            ThreadPool pool(std::thread::hardware_concurrency());
            size_t total_keys = publisher_list.size();

            // Calculate number of batches
            size_t batch_count = (total_keys + min_batch_size - 1) / min_batch_size;

            // Process in batches
            for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
                size_t start_index = batch_index * min_batch_size;
                size_t end_index = std::min(start_index + min_batch_size, total_keys);

                pool.Enqueue([&, start_index, end_index]() {
                    // Temporary storage for this batch's results.
                    std::unordered_map<NodeType, std::vector<NodeType>> batch_reachables;
                    std::unordered_map<NodeType, size_t> batch_counts;

                    // Process each node in batch
                    for (size_t i = start_index; i < end_index; ++i) {
                        NodeType key = publisher_list[i].first;
                        // Run BFS to find all reachable nodes
                        std::vector<NodeType> visited_vec = RunSimpleBFS(key);

                        batch_reachables[key] = std::move(visited_vec);
                        batch_counts[key] = batch_reachables[key].size();
                    }

                    // Merge results with global storage
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

            pool.WaitAll();  // Wait for all batches to complete
        }

        // Store final results.
        for (const auto& pair : local_reachables) {
            reachable_vec[pair.first] = { pair.second };
        }

        // Update publisher connection counts
        for (auto& entry : publisher_list) {
            if (local_counts.count(entry.first)) {
                entry.second = local_counts[entry.first];
            }
        }

        // Sort by connection count (descending) then by node value.
        std::sort(publisher_list.begin(), publisher_list.end(),
            [](const std::pair<NodeType, size_t>& first,
               const std::pair<NodeType, size_t>& second) {
                return first.second != second.second ?
                    first.second > second.second : first.first < second.first;
            });

        #ifdef DEBUG
            auto end_time = std::chrono::steady_clock::now();
            auto total_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            DEBUG_LOG("ComputeAllConnectionCounts completed in " << total_duration.count() << "ms");
            DEBUG_LOG("Processed " << publisher_list.size() << " nodes");
        #endif
    }

    /**
     * @brief Returns true if the graph contains no nodes
     */
    bool Empty() const {
        return adj_list.empty();
    }

    /**
     * @brief Clears all graph data and resets counters
     */
    void Clear() {
        adj_list.clear();
        reachable_vec.clear();
        publisher_list.clear();
        publisher_count = 0;
        node_count = 0;
    }

    /**
     * @brief Optimizes graph storage by preallocating memory
     */
    void Graph() {
        adj_list.max_load_factor(0.25);
        adj_list.reserve(2048);
        reachable_vec.max_load_factor(0.25);
        reachable_vec.reserve(2048);
        publisher_list.reserve(2048);
    }

    /**
     * @brief Returns the number of publisher nodes
     */
    size_t GetPubCount() const {
        return publisher_count;
    }

    /**
     * @brief Returns the total number of nodes in the graph
     */
    size_t GetnNodeCount() const {
        return node_count;
    }

    /**
     * @brief Returns the number of nodes with reachability data
     */
    size_t GetReachCount() const {
        return publisher_list.size();
    }

 private:
    /**
     * @brief Sorts publisher_list based on node values.
     * 
     * Uses different strategies based on list size:
     * - For small lists (<10,000): direct std::sort
     * - For large lists: manual split-sort-merge
     */
    void SortKeys() const {
        // Direct sorting for small lists (better cache locality)
        if (publisher_list.size() < 10000) {
            std::sort(publisher_list.begin(), publisher_list.end(),
                [](const auto& first, const auto& second) {
                    // Sort by node name (first element of pair)
                    return first.first < second.first;
                });
        } else {  // Manual split-sort-merge for large lists (better parallelism potential)
            // Split the list into two halves
            auto mid = publisher_list.begin() + publisher_list.size() / 2;
            std::vector<std::pair<NodeType, size_t>> left(publisher_list.begin(), mid);
            std::vector<std::pair<NodeType, size_t>> right(mid, publisher_list.end());

            // Sort each half independently
            std::sort(left.begin(), left.end(),
                [](const auto& first, const auto& second) {
                    return first.first < second.first;
                });

            std::sort(right.begin(), right.end(),
                [](const auto& first, const auto& second) {
                    return first.first < second.first;
                });

            // Merge the sorted halves back together
            std::merge(left.begin(), left.end(),
                       right.begin(), right.end(),
                       publisher_list.begin(),
                [](const auto& first, const auto& second) {
                    return first.first < second.first;
                });
        }
    }

    /**
     * @brief Performs BFS from a given source node.
     * 
     * @param source_node The starting node for BFS
     * @return Vector of nodes reachable from source_node (excluding the source itself)
     */
    std::vector<NodeType> RunSimpleBFS(const NodeType& source_node) {
        std::queue<NodeType> queue;            // Queue for BFS traversal
        std::unordered_set<NodeType> visited;  // Track visited nodes

        // Initialize with source node
        queue.push(source_node);
        visited.insert(source_node);

        // Standard BFS loop
        while (!queue.empty()) {
            // Get next node to process
            NodeType current = queue.front();
            queue.pop();

            // Find current node in adjacency list
            auto graph_node = adj_list.find(current);
            if (graph_node == adj_list.end()) continue;  // Skip if no edges

            // Process all neighbors
            const auto& neighbors = graph_node->second;
            for (const auto& neighbor_pair : neighbors) {
                const NodeType& neighbor = neighbor_pair.first;

                // Add to queue if not already visited
                if (visited.insert(neighbor).second) {
                    queue.push(neighbor);
                }
            }
        }

        // Remove source node from results (only want reachable nodes)
        visited.erase(source_node);

        // Convert set to vector for return
        return std::vector<NodeType>(visited.begin(), visited.end());
    }

    // Member variables with descriptions
    mutable std::unordered_map<NodeType, std::vector<std::pair<NodeType, float>>> adj_list;
    mutable std::unordered_map<NodeType, std::vector<std::vector<NodeType>>> reachable_vec;
    mutable std::vector<std::pair<NodeType, size_t>> publisher_list;
    mutable std::mutex log_mtx;  // Mutex for thread-safe logging

    size_t publisher_count = 0;  // Count of publisher nodes
    size_t node_count = 0;       // Total node count in graph
};

/**
 * @brief Reads student records from a binary file and constructs a directed graph
 * 
 * @tparam T Data type (must be StudentType)
 * @param file_number File number suffix (e.g., "1" for "pairs1.bin")
 * @param graph [out] Graph instance to be populated with the data
 */
template <typename T>
static void ReadBinary(const std::string& file_number, DirectedGraph<T>& graph) {
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
        graph.AddEdge(student.publisher, student.subscriber, student.weight);
    }
}

/**
 * @brief Task 1: Build adjacency lists from binary data and save results
 * 
 * 1. Reads student records from binary file
 * 2. Constructs directed graph adjacency lists
 * 3. Saves formatted results to output file
 * 
 * @param dir_graph Graph instance to operate on
 * @param file_number File number suffix for input/output files
 */
static void Task1(DirectedGraph<std::string>& dir_graph, const std::string& file_number) {
    ReadBinary(file_number, dir_graph);

    dir_graph.SaveToAdjFile(file_number);

    std::cout << "<<< There are " << dir_graph.GetPubCount() << " IDs in total. >>>\n\n"
              << "<<< There are " << dir_graph.GetnNodeCount() << " nodes in total. >>>\n";
}

/**
 * @brief Task 2: Compute connection counts using BFS and save results
 * 
 * 1. Uses parallel BFS to compute node reachability
 * 2. Counts connections for each node
 * 3. Saves sorted results to output file
 * 
 * @param dir_graph Graph instance with adjacency lists already built
 * @param file_number File number suffix for output file
 */
static void Task2(DirectedGraph<std::string>& dir_graph, const std::string& file_number) {
    dir_graph.ComputeAllConnectionCounts("max");
    dir_graph.SaveToCntFile(file_number);

    std::cout << "<<< There are " << dir_graph.GetReachCount() << " IDs in total. >>>\n";
}

/**
 * @brief Main program entry point with menu-driven interface
 * 
 * Provides interactive menu for graph operations:
 * 0. Quit program
 * 1. Build adjacency lists (Task 1)
 * 2. Compute connection counts (Task 2)
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
    DirectedGraph<std::string> dir_graph;

    dir_graph.Graph();  // Initialize graph with optimal settings

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
                dir_graph.Clear();  // Reset existing graph
            }

            if (file_number != "0") {
                struct stat buffer;
                file_name = "pairs" + file_number + ".bin";
                std::cout << '\n';

                // Check file existence
                if (stat(file_name.c_str(), &buffer)) {
                    std::cout << '\n';
                    std::cout << "### " << file_name << " does not exist! ###\n";
                } else {
                    // Execute Task 1
                    Task1(dir_graph, file_number);
                }
            }
            std::cout << '\n';

            break;
        case 2:
            if (dir_graph.Empty()) {
                std::cout << "### There is no graph and choose 1 first. ###\n";
            } else {
                // Execute Task 2
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
