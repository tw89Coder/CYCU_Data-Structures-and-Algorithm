/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex03_27_10927262.cpp
 * @brief A program that implements hash tables with linear probing and double hashing to efficiently manage and organize graduate student data.
 * @version 1.0.3
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

#pragma GCC optimize("Ofast")
// C++ Standard Library (alphabetical order)
#include <algorithm>   // Algorithms (sort, find, transform)
#include <atomic>      // Atomic operations and thread synchronization
#include <fstream>     // File stream operations (ifstream, ofstream)
#include <iomanip>     // I/O formatting (setw, setprecision)
#include <iostream>    // Standard I/O streams (cin, cout)
#include <limits>      // Numeric limits (numeric_limits)
#include <mutex>       // Mutual exclusion (mutex, lock_guard)
#include <numeric>     // Numeric operations (accumulate, inner_product)
#include <sstream>     // String streams (istringstream, ostringstream)
#include <stdexcept>   // Standard exceptions (runtime_error, logic_error)
#include <string>      // String class and operations
#include <thread>      // Thread support (std::thread)
#include <vector>      // Dynamic array container

// C Standard Library (C++ wrapper headers, alphabetical order)
#include <cstddef>     // Fundamental types (size_t, nullptr_t)
#include <cstdint>     // Fixed-width integer types (int32_t, uint64_t)
#include <cstdlib>     // General utilities (malloc, exit, atoi)
#include <cstring>     // C-style string operations (strcpy, memcmp)

#define COLUMNS 6     // Number of scores for each student.
#define MAX_LEN 10    // Array size of student id and name.
#define BIG_INT 255   // Integer upper bound.

#ifdef DEBUG
    // Thread-safe debug logging
    #define DEBUG_LOG(msg) { std::lock_guard<std::mutex> lock(log_mtx); \
                             std::cout << "[DEBUG] " << msg << std::endl; }
#else
    // Expands to nothing in non-DEBUG builds (no runtime overhead) 
    #define DEBUG_LOG(msg)   
#endif

// Sieve of Eratosthenes data structures:
static std::vector<bool> is_prime;   // Flags indicating primality (true = prime)
static std::vector<int> prime_list;  // List of actual prime numbers found
static int prime_limit = 1;          // Current upper bound for generated primes

// Forward declaration
static int FindNextPrimeAbove(double var);

/**
 * @struct StudentType
 * @brief Represents student record data.
 */
typedef struct st {
    char sid[MAX_LEN];             // student id
    char sname[MAX_LEN];           // student name
    unsigned char score[COLUMNS];  // A set of scores in [0, 100]
    float mean;                    // The average of scores.
} StudentType;

/**
 * @class HashTable
 * @brief Implements hash tables with linear probing and double hashing techniques.
 *
 * This class provides functionality to store and manage graduate student data using
 * either linear probing or double hashing collision resolution methods. It supports
 * insertion, searching, and performance analysis operations.
 */
class HashTable {
 public:
    /**
     * @enum Mode
     * @brief Specifies the hashing technique to be used.
     *
     * Linear probing collision resolution.
     * Double hashing collision resolution.
     */
    enum class Mode { LINEAR, DOUBLE };

    /**
     * @brief Constructs a HashTable with specified mode and size.
     * @param mode_str The hashing mode ("linear" or "double").
     * @param table_size The initial size of the hash table.
     * @throw std::invalid_argument If an unsupported mode is provided.
     */
    HashTable(const std::string& mode_str, const size_t& table_size)
        : table_size_(table_size), table_(table_size), occupied_(table_size, false) {
        // Initialize hash table based on specified mode.
        if (mode_str == "linear") {
            mode_ = Mode::LINEAR;
        } else if (mode_str == "double") {
            mode_ = Mode::DOUBLE;
        } else {
            throw std::invalid_argument("Unsupported hash mode: " + mode_str);
        }
    }

    /**
     * @brief Inserts a student record into the hash table.
     * @param student The student data to insert.
     * @param info_num The total number of student records (used for step calculation).
     * @return true if insertion was successful, false otherwise.
     */
    bool Insert(const StudentType& student, const size_t& info_num) {
        // Calculate initial hash position.
        int index = Hash(student.sid);

        // Determine step size based on hashing mode.
        double target = static_cast<double>(info_num) / 5;
        size_t max_step = FindNextPrimeAbove(target);
        int step = (mode_ == Mode::DOUBLE) ? Step(student.sid, max_step) : 1;

        // Prepare hash entry for insertion.
        HashEntry temp_entry;
        temp_entry.hvalue = index % table_size_;
        std::strncpy(temp_entry.sid, student.sid, MAX_LEN - 1);
        temp_entry.sid[MAX_LEN - 1] = '\0';
        std::strncpy(temp_entry.sname, student.sname, MAX_LEN - 1);
        temp_entry.sname[MAX_LEN - 1] = '\0';
        temp_entry.mean = student.mean;

        // Find empty slot using probing.
        for (int i = 0; i < table_size_; ++i) {
            int try_index = (index + i * step) % table_size_;
            if (!occupied_[try_index]) {
                table_[try_index] = temp_entry;
                occupied_[try_index] = true;
                return true;
            }
        }

        // If no empty slot found.
        std::cerr << "HashTable full. Cannot insert " << student.sid << "\n";
        return false;
    }

    /**
     * @brief Calculates the average comparisons for unsuccessful searches.
     * @param info_num The total number of student records.
     * @return The average number of comparisons needed.
     *
     * This method uses multiple threads to parallelize the computation.
     */
    double UnsuccessfulSearch(const size_t& info_num) const {
        // Setup for parallel processing.
        double target = static_cast<double>(info_num) / 5.0;
        size_t max_step = (mode_ == Mode::DOUBLE) ? FindNextPrimeAbove(target) : 1;

        size_t thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 4;

        std::vector<std::thread> threads;
        std::vector<double> partial_sums(thread_count, 0.0);
        std::atomic<size_t> global_index(0);  // Dynamic Work Index. For Work Stealing.

        auto worker = [&](size_t tid) {
            DEBUG_LOG("Thread " << tid << " started");

            double local_sum = 0.0;

            while (true) {
                size_t i = global_index.fetch_add(1);  // Dynamic Index Allocation.
                if (i >= table_size_) break;

                if (!occupied_[i]) continue;

                // Calculate probing sequence
                size_t step = (mode_ == Mode::DOUBLE) ? Step(table_[i].sid, max_step) : 1;
                size_t index = (i + step) % table_size_;
                size_t count = 0;

                // Count comparisons until empty slot found
                while (index != i) {
                    ++count;
                    if (!occupied_[index]) {
                        local_sum += count;
                        DEBUG_LOG("Thread " << tid << " found empty slot at index " << index);
                        break;
                    }
                    index = (index + step) % table_size_;
                }

                if (index == i) {
                    DEBUG_LOG("Thread " << tid << " encountered an infinite loop at index " << i);
                }
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                partial_sums[tid] = local_sum;
            }

            DEBUG_LOG("Thread " << tid << " completed processing");
        };

        // Launch and join threads
        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back(worker, i);
        }

        for (auto& th : threads) {
            th.join();
        }

        // Calculate average comparisons
        double total_times = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
        return (total_times > 0) ? total_times / table_size_ : 0.0;
    }

    /**
     * @brief Calculates the average comparisons for successful searches.
     * @param info_num The total number of student records.
     * @return The average number of comparisons needed.
     *
     * This method uses multiple threads to parallelize the computation.
     */
    double SuccessfulSearch(const size_t& info_num) const {
        int totalComparisons = 0;
        int successfulSearches = 0;
        size_t thread_count = std::thread::hardware_concurrency();  // Get the number of available CPU threads.
        if (thread_count == 0) thread_count = 4;  // The default minimum number of threads is 4.

        std::vector<std::thread> threads;
        std::vector<int> localComparisons(thread_count, 0);
        std::vector<int> localSearches(thread_count, 0);
        std::atomic<size_t> global_index(0);  // Dynamic Work Index. For Work Stealing.

        auto search_task = [&](size_t tid) {
            DEBUG_LOG("Thread " << tid << " started");

            int localComparisonCount = 0;
            int localSearchCount = 0;

            while (true) {
                size_t start = global_index.fetch_add(1);  // Dynamic Index Allocation.
                if (start >= table_.size()) break;

                if (table_[start].sid[0] != '\0') {
                    int comparisons = SuccessfulSearchHelper(table_[start].sid, info_num);
                    localComparisonCount += comparisons;
                    ++localSearchCount;

                    DEBUG_LOG("Thread " << tid << " searched sid " << table_[start].sid
                              << ", comparisons: " << comparisons);
                }
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                localComparisons[tid] = localComparisonCount;
                localSearches[tid] = localSearchCount;
            }

            DEBUG_LOG("Thread " << tid << " completed, total comparisons: " << localComparisonCount
                      << ", successful searches: " << localSearchCount);
        };

        // Launch and join threads
        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back(search_task, i);
        }

        for (auto& th : threads) {
            th.join();
        }

        // Calculate average comparisons
        totalComparisons = std::accumulate(localComparisons.begin(), localComparisons.end(), 0);
        successfulSearches = std::accumulate(localSearches.begin(), localSearches.end(), 0);

        DEBUG_LOG("All threads completed, final total comparisons: " << totalComparisons
                  << ", final successful searches: " << successfulSearches);

        return (successfulSearches > 0) ? static_cast<double>(totalComparisons) / info_num : 0.0;
    }

    /**
     * @brief Prints the contents of the hash table to standard output.
     */
    void Print() const {
        std::cout << "\n[Hash Table - " << ((mode_ == Mode::LINEAR) ? "Linear" : "Double")
                    << " Hashing, Size: " << table_size_ << "]\n";
        for (int i = 0; i < table_size_; ++i) {
            std::cout << "[" << std::setw(3) << i << "]: ";
            if (occupied_[i]) {
                std::cout << std::setw(10) << table_[i].hvalue << ", "
                          << std::setw(10) << table_[i].sid << ", "
                          << std::setw(4)  << table_[i].sname << ", "
                          << std::setw(10) << table_[i].mean << "\n";
            } else {
                std::cout << "\n";
            }
        }

        std::cout << "-----------------------------------------------------\n";
    }

    /**
     * @brief Saves the hash table contents to a file.
     * @param file_name The name of the file to save to.
     */
    void SaveToFile(const std::string& file_name) const {
        std::ofstream out_file(file_name);
        if (!out_file.is_open()) {
            std::cerr << "Error: Can not open output file: " << file_name << "\n";
            return;
        }

        switch (mode_) {
            case Mode::LINEAR:
                out_file << " --- Hash table created by Linear probing    ---\n";

                break;
            case Mode::DOUBLE:
                out_file << " --- Hash table created by Double hashing    ---\n";

                break;
            default:
                out_file << " --- Hash table created by XXXXXX probing    ---\n";

                break;
        }

        for (int i = 0; i < table_size_; ++i) {
            out_file << "[" << std::setw(3) << i << "] ";
            if (occupied_[i]) {
                out_file << std::setw(10) << table_[i].hvalue << ", "
                         << std::setw(10) << table_[i].sid << ", "
                         << std::setw(10)  << table_[i].sname << ", "
                         << std::setw(10) << table_[i].mean << "\n";
            } else {
                out_file << "\n";
            }
        }

        out_file << " ----------------------------------------------------- \n";
        out_file.close();
    }

    /**
     * @brief Clears all entries from the hash table.
     */
    void Clear() {
        std::fill(occupied_.begin(), occupied_.end(), false);
    }

 private:
    /**
     * @struct HashEntry
     * @brief Represents an entry in the hash table.
     */
    struct HashEntry {
        int hvalue;                 // Hash value of the entry
        char sid[MAX_LEN];          // Student ID
        char sname[MAX_LEN];        // Student name
        float mean;                 // Average score
    };

    Mode mode_;                     // Hashing technique mode
    size_t table_size_;             // Size of the hash table
    std::vector<HashEntry> table_;  // Storage for hash table entries
    std::vector<bool> occupied_;    // Tracks occupied slots
    mutable std::mutex mtx;         // Mutex for thread safety
    mutable std::mutex log_mtx;     // Mutex for logging

    /**
     * @brief Computes the hash value for a given key.
     * @param key The string to hash (student ID).
     * @return The computed hash value.
     */
    size_t Hash(const char* key) const {
        if (!key) return 0;

        uint64_t  product = 1;
        for (int i = 0; key[i] != '\0'; ++i) {
            product *= static_cast<uint64_t >(key[i]);
        }

        return product % table_size_;
    }

    /**
     * @brief Computes the step size for double hashing.
     * @param key The string to use for step calculation.
     * @param max_step The maximum step size allowed.
     * @return The computed step size.
     */
    size_t Step(const char* key, const size_t& max_step) const {
        if (!key) return 0;

        uint64_t  product = 1;
        for (int i = 0; key[i] != '\0'; ++i) {
            product *= static_cast<uint64_t >(key[i]);
        }

        return max_step - (product % max_step);
    }

    /**
     * @brief Helper function for successful search operation.
     * @param sid The student ID to search for.
     * @param info_num Total number of student records.
     * @return Number of comparisons made during search.
     */
    int SuccessfulSearchHelper(const std::string& sid, const size_t& info_num) const {
        int index = Hash(sid.c_str());
        double target = static_cast<double>(info_num) / 5.0;
        size_t max_step = FindNextPrimeAbove(target);
        int step = (mode_ == Mode::DOUBLE) ? Step(sid.c_str(), max_step) : 1;
        int comparisons = 0;

        for (int i = 0; i < table_size_; ++i) {
            int try_index = (index + i * step) % table_size_;
            ++comparisons;

            if (occupied_[try_index] && std::strcmp(table_[try_index].sid, sid.c_str()) == 0) {
                return comparisons;  // Number of successful search comparisons.
            }
            if (!occupied_[try_index]) break;
        }

        return comparisons;
    }
};  // class HashTable

/**
 * @brief Generates prime numbers up to a specified limit using Sieve of Eratosthenes.
 * 
 * Maintains a cache of generated primes to optimize subsequent calls.
 * Updates the global prime_list and is_prime vectors.
 * 
 * @param new_limit The upper bound for prime number generation.
 */
static void GeneratePrimesUpTo(double new_limit) {
    if (new_limit <= prime_limit) return;  // Already generated, no need to repeat.

    if (prime_limit < 2) {
        prime_limit = 2;
        is_prime.assign(new_limit + 1, true);
        is_prime[0] = is_prime[1] = false;
        prime_list.clear();
    } else {
        is_prime.resize(new_limit + 1, true);
    }

    for (int i = 2; i * i <= new_limit; ++i) {
        if (is_prime[i]) {
            int start = std::max(i * i, ((prime_limit + i - 1) / i) * i);
            for (int j = start; j <= new_limit; j += i) {
                is_prime[j] = false;
            }
        }
    }

    for (int i = prime_limit + 1; i <= new_limit; ++i) {
        if (is_prime[i]) {
            prime_list.push_back(i);
        }
    }

    prime_limit = new_limit;
}

/**
 * @brief Finds the smallest prime number greater than the target value.
 * 
 * Uses pre-generated primes if available, otherwise generates new primes as needed.
 * 
 * @param target The value to find the next prime above.
 * @return int The smallest prime greater than target, or -1 if error occurs.
 */
static int FindNextPrimeAbove(double target) {
    GeneratePrimesUpTo(target * 2);  // Reserve space

    auto it = std::lower_bound(prime_list.begin(), prime_list.end(), target);
    if (it != prime_list.end()) {
        return *it;
    }

    // If prime_list is not large enough, expand it and find
    GeneratePrimesUpTo(target * 4);
    it = std::lower_bound(prime_list.begin(), prime_list.end(), target);
    return (it != prime_list.end()) ? *it : -1;  // -1 means not found (theoretically not)
}

/**
 * @brief Checks if a file exists in the filesystem.
 * 
 * @param filename The name/path of the file to check.
 * @return true if file exists and is accessible, false otherwise.
 */
static bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

/**
 * @brief Converts a text file containing student records to a binary format.
 * 
 * Reads a tab-delimited text file with student data and writes it in binary format.
 * The text file should contain student ID, name, scores, and mean in each line.
 * 
 * @param output_file_name [out] Will contain the name of the created binary file.
 * @param file_number [in,out] The number used to identify input/output files.
 * @return int Number of student records converted, or 0 if operation failed.
 */
static int Text2Binary(std::string& output_file_name, std::string& file_number) {
    std::fstream in_file, out_file;
    int student_count = 0;

    while (true) {
        std::cout << "Input a file number ([0] Quit): ";
        std::cin >> file_number;
        std::cout << std::endl;
        if (file_number == "0") {
            return 0;
        }

        output_file_name = "input" + file_number + ".bin";

        if (fileExists(output_file_name)) {
            return 1;
        } else {
            std::cout << "### " << output_file_name << " does not exist! ###\n";
            std::cout << std::endl;
        }

        in_file.open("input" + file_number + ".txt", std::ios::in);
        if (in_file.is_open()) {
            break;
        }

        std::cout << "### " << ("input" + file_number + ".txt") << " does not exist! ###\n";
        std::cout << std::endl;

        return 0;
    }

    output_file_name = "input" + file_number + ".bin";
    out_file.open(output_file_name, std::ios::out | std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "Cannot create output binary file.\n";
        in_file.close();
        return 0;
    }

    char read_buffer[BIG_INT];
    while (in_file.getline(read_buffer, BIG_INT)) {
        std::string line(read_buffer);
        StudentType student = {};
        std::size_t start = 0;
        std::size_t tab_pos = 0;
        int column_number = 0;

        while ((tab_pos = line.find('\t', start)) != std::string::npos) {
            std::string field = line.substr(start, tab_pos - start);
            switch (column_number) {
                case 0:
                    strncpy(student.sid, field.c_str(), MAX_LEN - 1);
                    student.sid[MAX_LEN - 1] = '\0';  // Ensure string termination
                    break;
                case 1:
                    strncpy(student.sname, field.c_str(), MAX_LEN - 1);
                    student.sname[MAX_LEN - 1] = '\0';  // Ensure string termination
                    break;
                default:
                    if (column_number - 2 < COLUMNS) {
                        student.score[column_number - 2] =
                            static_cast<unsigned char>(std::atoi(field.c_str()));
                    }
                    break;
            }
            ++column_number;
            start = tab_pos + 1;
        }

        // Last field is mean
        if (start < line.length()) {
            student.mean = static_cast<float>(std::atof(line.substr(start).c_str()));
        }

        out_file.write(reinterpret_cast<const char*>(&student), sizeof(student));
        ++student_count;
    }

    std::cout << "---" + output_file_name + " has been created ---\n";
    std::cout << std::endl;

    in_file.close();
    out_file.close();
    return student_count;
}

/**
 * @brief Reads student records from a binary file into a vector.
 * 
 * @tparam T The type of data to read (should be StudentType).
 * @param file_name Name of the binary file to read from.
 * @param vec [out] Vector that will contain the read student records.
 */
template <typename T>
static void ReadBinary(const std::string& file_name, std::vector<T>& vec) {
    std::fstream binary_file(file_name, std::ios::in | std::ios::binary);
    if (!binary_file.is_open()) {
        std::cerr << "Cannot open binary file: " << file_name << "\n";
        return;
    }

    StudentType student = {};

    while (binary_file.read(reinterpret_cast<char*>(&student), sizeof(student))) {
        vec.push_back(student);
    }

    binary_file.close();
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
static void Task1(const std::vector<StudentType>& student_info, const std::string& file_number) {
    double target = student_info.size() * 1.1;
    size_t hash_size = FindNextPrimeAbove(target);
    std::string file_name = "linear" + file_number + ".txt";
    std::ostringstream buffer;

    HashTable X("linear", hash_size);

    for (const auto& student : student_info) {
        if (!X.Insert(student, student_info.size())) {
            std::cerr << "Can not INSERT: " << student.sid << "\n";
        }
    }

    buffer << "Hash table has been successfully created by Linear probing   \n";

    X.SaveToFile(file_name);
    buffer << std::fixed << std::setprecision(4);
    buffer << "unsuccessful search: " << X.UnsuccessfulSearch(student_info.size())
           << " comparisons on average\n";
    buffer << "successful search: " << X.SuccessfulSearch(student_info.size())
           << " comparisons on average\n";

    std::cout << buffer.str();
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
static void Task2(const std::vector<StudentType>& student_info, const std::string& file_number) {
    double target = student_info.size() * 1.1;
    size_t hash_size = FindNextPrimeAbove(target);
    std::string file_name = "double" + file_number + ".txt";
    std::ostringstream buffer;

    HashTable Y("double", hash_size);

    for (const auto& student : student_info) {
        if (!Y.Insert(student, student_info.size())) {
            std::cerr << "Can not INSERT: " << student.sid << "\n";
        }
    }

    buffer << "Hash table has been successfully created by Double hashing   \n";

    Y.SaveToFile(file_name);
    buffer << std::fixed << std::setprecision(4);
    buffer << "successful search: " << Y.SuccessfulSearch(student_info.size())
           << " comparisons on average\n";

    std::cout << buffer.str();
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
    int select_command = 0;
    std::string file_name;
    std::string file_number;
    std::vector<StudentType> temp_info;
    std::ostringstream buffer;

    do {
        while (true) {
            // Display the menu options for the user
            std::cout <<
                "******* Hash Table *****\n"
                "* 0. QUIT              *\n"
                "* 1. Linear probing    *\n"
                "* 2. Double hashing    *\n"
                "************************\n"
                "Input a choice(0, 1, 2): ";

            std::cin >> select_command;

            // Check if the input is valid
            if (!std::cin.fail()) {
                break;
            } else {
                // Handle invalid input
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                buffer << std::endl;
                buffer << "Command does not exist!\n";
                buffer << std::endl;
                buffer << std::endl;

                std::cout << buffer.str();
                buffer.str("");
                buffer.clear();
            }
        }

        // Handle the different options based on the user's choice
        switch (select_command) {
        case 0:
            break;
        case 1:
            std::cout << std::endl;

            if (Text2Binary(file_name, file_number) > 0) {
                if (!temp_info.empty()) {
                    temp_info.clear();
                }

                ReadBinary(file_name, temp_info);
                Task1(temp_info, file_number);
            }

            std::cout << std::endl;

            break;
        case 2:
            if (temp_info.empty()) {
                std::cout << "### Command 1 first. ###\n";
                std::cout << std::endl;
            } else {
                std::cout << std::endl;
                Task2(temp_info, file_number);
            }
            std::cout << std::endl;

            break;
        default:
            buffer << std::endl;
            buffer << "Command does not exist!\n";
            buffer << std::endl;
            buffer << std::endl;

            std::cout << buffer.str();
            buffer.str("");
            buffer.clear();
        }
    } while (select_command != 0);  // Continue until the user selects option 0 (quit)

    return 0;
}
