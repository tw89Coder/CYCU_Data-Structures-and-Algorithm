/**
 * @copyright 2025 Group 27. All rights reserved.
 * @file DS2ex03_27_10927262.cpp
 * @brief A program that implements hash tables with linear probing and double hashing to efficiently manage and organize graduate student data.
 * @version 1.0.1
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
 #include <algorithm>
 #include <atomic>
 #include <cstddef>
 #include <cstdlib>
 #include <cstring>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <limits>
 #include <mutex>
 #include <numeric>
 #include <stdexcept>
 #include <string>
 #include <sstream>
 #include <thread>
 #include <vector>

#define COLUMNS 6    // Number of scores for each student.
#define MAX_LEN 10   // Array size of student id and name.
#define BIG_INT 255  // Integer upper bound.

#ifdef DEBUG
    #define DEBUG_LOG(msg) { std::lock_guard<std::mutex> lock(log_mtx); std::cout << "[DEBUG] " << msg << std::endl; }
#else
    #define DEBUG_LOG(msg)
#endif


static std::vector<bool> is_prime;
static std::vector<int> prime_list;
static int prime_limit = 1;

// Forward declaration
static int FindNextPrimeAbove(double var);

typedef struct st {
    char sid[MAX_LEN];             // student id
    char sname[MAX_LEN];           // student name
    unsigned char score[COLUMNS];  // A set of scores in [0, 100]
    float mean;                    // The average of scores.
} StudentType;

class HashTable {
 public:
    enum class Mode { LINEAR, DOUBLE };

    HashTable(const std::string& mode_str, const size_t& table_size)
        : table_size_(table_size), table_(table_size), occupied_(table_size, false) {
        if (mode_str == "linear") {
            mode_ = Mode::LINEAR;
        } else if (mode_str == "double") {
            mode_ = Mode::DOUBLE;
        } else {
            throw std::invalid_argument("Unsupported hash mode: " + mode_str);
        }
    }

    bool Insert(const StudentType& student, const size_t& info_num) {
        int index = Hash(student.sid);
        double target = (double)info_num / 5;
        size_t max_step = FindNextPrimeAbove(target);
        int step = (mode_ == Mode::DOUBLE) ? Step(student.sid, max_step) : 1;

        HashEntry temp_entry;
        temp_entry.hvalue = index % table_size_;
        std::strncpy(temp_entry.sid, student.sid, MAX_LEN - 1);
        temp_entry.sid[MAX_LEN - 1] = '\0';
        std::strncpy(temp_entry.sname, student.sname, MAX_LEN - 1);
        temp_entry.sname[MAX_LEN - 1] = '\0';
        temp_entry.mean = student.mean;

        for (int i = 0; i < table_size_; ++i) {
            int try_index = (index + i * step) % table_size_;
            if (!occupied_[try_index]) {
                table_[try_index] = temp_entry;
                occupied_[try_index] = true;
                return true;
            }
        }
        std::cerr << "HashTable full. Cannot insert " << student.sid << "\n";
        return false;
    }

    double UnsuccessfulSearch(const size_t& info_num) const {
        double target = static_cast<double>(info_num) / 5.0;
        size_t max_step = (mode_ == Mode::DOUBLE) ? FindNextPrimeAbove(target) : 1;
        
        size_t thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 4;
    
        std::vector<std::thread> threads;
        std::vector<double> partial_sums(thread_count, 0.0);
        std::atomic<size_t> global_index(0); // 動態工作索引
    
        auto worker = [&](size_t tid) {
            DEBUG_LOG("Thread " << tid << " started");
    
            double local_sum = 0.0;
    
            while (true) {
                size_t i = global_index.fetch_add(1); // 動態分配索引
                if (i >= table_size_) break;
    
                if (!occupied_[i]) continue;
    
                size_t step = (mode_ == Mode::DOUBLE) ? Step(table_[i].sid, max_step) : 1;
                size_t index = (i + step) % table_size_;
                size_t count = 0;
    
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
    
        // 啟動執行緒
        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back(worker, i);
        }
    
        for (auto& th : threads) {
            th.join();
        }
    
        double total_times = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
        return (total_times > 0) ? total_times / table_size_ : 0.0;
    }

    double SuccessfulSearch(const size_t& info_num) const {
        int totalComparisons = 0;
        int successfulSearches = 0;
        size_t thread_count = std::thread::hardware_concurrency(); // 取得可用 CPU 執行緒數量
        if (thread_count == 0) thread_count = 4; // 預設最少 4 個執行緒
        
        std::vector<std::thread> threads;
        std::vector<int> localComparisons(thread_count, 0);
        std::vector<int> localSearches(thread_count, 0);
        std::atomic<size_t> global_index(0); // 動態工作索引
    
        auto search_task = [&](size_t tid) {
            DEBUG_LOG("Thread " << tid << " started");
            
            int localComparisonCount = 0;
            int localSearchCount = 0;
    
            while (true) {
                size_t start = global_index.fetch_add(1); // 動態分配索引
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
    
        // 啟動執行緒
        for (size_t i = 0; i < thread_count; ++i) {
            threads.emplace_back(search_task, i);
        }
    
        for (auto& th : threads) {
            th.join(); // 確保所有執行緒都完成
        }
    
        // 計算總比對次數和成功搜尋次數
        totalComparisons = std::accumulate(localComparisons.begin(), localComparisons.end(), 0);
        successfulSearches = std::accumulate(localSearches.begin(), localSearches.end(), 0);
    
        DEBUG_LOG("All threads completed, final total comparisons: " << totalComparisons
                  << ", final successful searches: " << successfulSearches);
    
        return (successfulSearches > 0) ? static_cast<double>(totalComparisons) / info_num : 0.0;
    }
    

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

    void SaveToFile(const std::string& file_name) const {
        std::ofstream out_file(file_name);
        if (!out_file.is_open()) {
            std::cerr << "Error: 無法開啟輸出文件 " << file_name << "\n";
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

    void Clear() {
        std::fill(occupied_.begin(), occupied_.end(), false);
    }

 private:
    struct HashEntry {
        int hvalue;
        char sid[MAX_LEN];
        char sname[MAX_LEN];
        float mean;
    };  

    Mode mode_;
    size_t table_size_;
    std::vector<HashEntry> table_;
    std::vector<bool> occupied_;
    mutable std::mutex mtx;
    mutable std::mutex log_mtx;

    size_t Hash(const char* key) const {
        if (!key) return 0;
    
        unsigned long long product = 1;
        for (int i = 0; key[i] != '\0'; ++i) {
            product *= static_cast<unsigned long long>(key[i]);
        }

        return product % table_size_;
    }

    size_t Step(const char* key, const size_t& max_step) const {
        if (!key) return 0;
    
        unsigned long long product = 1;
        for (int i = 0; key[i] != '\0'; ++i) {
            product *= static_cast<unsigned long long>(key[i]);
        }

        return max_step - (product % max_step);
    }

    int SuccessfulSearchHelper(const std::string& sid, const size_t& info_num) const {
        int index = Hash(sid.c_str());
        double target = (double)info_num / 5;
        size_t max_step = FindNextPrimeAbove(target);
        int step = (mode_ == Mode::DOUBLE) ? Step(sid.c_str(), max_step) : 1;
        int comparisons = 0;

        for (int i = 0; i < table_size_; ++i) {
            int try_index = (index + i * step) % table_size_;
            ++comparisons;

            if (occupied_[try_index] && std::strcmp(table_[try_index].sid, sid.c_str()) == 0) {
                return comparisons;  // 成功搜尋比較次數
            }
            if (!occupied_[try_index]) break; 
        }

        return comparisons; 
    }
};    
    
// 使用埃拉托色尼篩法建立質數表
static void GeneratePrimesUpTo(double new_limit) {
    if (new_limit <= prime_limit) return;  // 已生成過，無需重複

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

static int FindNextPrimeAbove(double target) {
    GeneratePrimesUpTo(target * 2);  // 預留空間

    auto it = std::lower_bound(prime_list.begin(), prime_list.end(), target);
    if (it != prime_list.end()) {
        return *it;
    }

    // 若 prime_list 沒有足夠大，擴充再找
    GeneratePrimesUpTo(target * 4);
    it = std::lower_bound(prime_list.begin(), prime_list.end(), target);
    return (it != prime_list.end()) ? *it : -1;  // -1 表示沒找到（理論上不會）
}


static bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

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

template <typename T>
static void ReadBinary(const std::string& file_name, std::vector<T>& vec) {
    std::fstream binary_file(file_name, std::ios::in | std::ios::binary);
    if (!binary_file.is_open()) {
        std::cerr << "Cannot open binary file: " << file_name << "\n";
        return;
    }

    StudentType student = {};
    int index = 0;

    while (binary_file.read(reinterpret_cast<char*>(&student), sizeof(student))) {
        vec.push_back(student);
    }

    binary_file.close();
}

static void Task1(const std::vector<StudentType>& student_info, const std::string& file_number) {
    double target = student_info.size() * 1.1;
    size_t hash_size = FindNextPrimeAbove(target);
    std::string file_name = "linear" + file_number + ".txt";
    std::ostringstream buffer;

    HashTable X("linear", hash_size);

    for (const auto& student : student_info) {
        if (!X.Insert(student, student_info.size())) {
            std::cerr << "無法插入：" << student.sid << "\n";
        }
    }

    buffer << "Hash table has been successfully created by Linear probing   \n";

    X.SaveToFile(file_name);
    buffer << std::fixed << std::setprecision(4);
    buffer << "unsuccessful search: " << X.UnsuccessfulSearch(student_info.size()) << " comparisons on average\n";
    buffer << "successful search: " << X.SuccessfulSearch(student_info.size()) << " comparisons on average\n";

    std::cout << buffer.str();
}

static void Task2(const std::vector<StudentType>& student_info, const std::string& file_number) {
    double target = student_info.size() * 1.1;
    size_t hash_size = FindNextPrimeAbove(target);
    std::string file_name = "double" + file_number + ".txt";
    std::ostringstream buffer;

    HashTable Y("double", hash_size);

    for (const auto& student : student_info) {
        if (!Y.Insert(student, student_info.size())) {
            std::cerr << "無法插入：" << student.sid << "\n";
        }
    }

    buffer << "Hash table has been successfully created by Double hashing   \n";

    Y.SaveToFile(file_name);
    buffer << std::fixed << std::setprecision(4);
    buffer << "successful search: " << Y.SuccessfulSearch(student_info.size()) << " comparisons on average\n";

    std::cout << buffer.str();
}

/**
 * @brief Main function to drive the program logic.
 * 
 * The main function provides a menu for the user to select operations. It handles task selection, builds the appropriate tree
 * (2-3 tree or AVL tree), and displays relevant information about the trees. 
 * The user can keep choosing options until they select to quit (option 0).
 * 
 * @return int Returns 0 on successful execution.
 */
int main() {
    int select_command = 0;
    int select_lower_bound = 0;
    int select_upper_bound = 2;
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
