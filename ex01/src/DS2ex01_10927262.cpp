/** 
 * @file DS2ex01_10927262.cpp
 * @brief A program that uses Heap to manage graduate data.
 * @version 2.1.0
 *
 * @details
 * This program reads and processes graduate data using a heap-based structure.
 * It supports min-heaps, max-heaps, and deaps for efficient data handling.
 *
 * @author 
 * - 10927262 呂易鴻
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @class HeapBase
 * @brief Base class for heap structures.
 *
 * This class provides common heap functionalities, including insertion, deletion,
 * height calculation, and heap property maintenance.
 */
template <typename T>
class HeapBase {
protected:
    std::vector<std::pair<int, T>> data;

    virtual void heapifyUp(int index) = 0;  // pure virtual
    virtual void heapifyDown(int index) = 0;

public:
    virtual ~HeapBase() = default;

    /** @brief Checks if the heap is empty. */
    bool isEmpty() const { return data.empty(); }

    /** @brief Returns the size of the heap. */
    int size() const { return data.size(); }

    /** @brief Clears all elements from the heap. */
    void clear() { data.clear(); }

    /** @brief Returns the top (root) element of the heap. */
    const std::pair<int, T>& top() const {
        if (data.empty()) throw std::runtime_error("Heap is empty!");

        return data[0];
    }

    /** @brief Returns the bottom-most element of the heap. */
    const std::pair<int, T>& bottom() const {
        if (data.empty()) throw std::runtime_error("Heap is empty!");

        return data.back();
    }

    /** @brief Returns the leftmost bottom element of the heap. */
    const std::pair<int, T>& leftmostBottom() const {
        if (data.empty()) throw std::runtime_error("Heap is empty!");
    
        int index = 0;
        while (2 * index + 1 < data.size()) {
            index = 2 * index + 1;  // Move to the left node.
        }
    
        return data[index];
    }    

    /** @brief Inserts a new element into the heap. */
    virtual void push(int key, const T& value) {
        data.emplace_back(key, value);
        heapifyUp(data.size() - 1);
    }

    /** @brief Removes and returns the top element of the heap. */
    virtual std::pair<int, T> pop() {
        if (data.empty()) throw std::runtime_error("Heap is empty!");
        std::pair<int, T> popData = data[0];
        data[0] = data.back();
        data.pop_back();

        if (!data.empty()) heapifyDown(0);

        return popData;
    }

    /** @brief Returns the height of the heap. */
    int height() const {
        if (data.empty()) return 0;

        return static_cast<int>(std::floor(std::log2(data.size()))) + 1;
    }

    /** @brief Returns the height of the heap including additional nodes. */
    int height(int nodeNum) const {
        if (data.empty()) return 0;

        return static_cast<int>(std::floor(std::log2(data.size() + nodeNum))) + 1;
    }

    /** @brief Replaces an element at a specific index and restores heap property. */
    std::pair<int, T> replaceAt(int index, int newKey, const T& newValue) {
        if (index < 0 || (index >= data.size())) {
            throw std::out_of_range("Index out of range");
        }

        // Store the original value before replacing
        std::pair<int, T> oldValue = data[index];

        // Replace the element at the specified index
        data[index] = std::make_pair(newKey, newValue);

        // Heapify up and down to maintain heap property
        heapifyUp(index);

        // Return the original value
        return oldValue;
    }

    /** @brief Retrieves an element at a specific index. */
    const std::pair<int, T>& getAt(int index) const {
        if (index < 0 || (index >= data.size())) {
            throw std::out_of_range("Index out of range");
        }

        return data[index];
    }

    /** @brief Checks if the heap is a full binary tree. */
    bool isFullBinaryTree(int nodeCount) const {
        if (nodeCount == 0) return false;
    
        int height = static_cast<int>(std::log2(nodeCount + 1));
    
        return nodeCount == (1 << height) - 1;
    }

    /** @brief Prints the heap elements. */
    void printHeap() const {
        for (const auto& item : data) {
            std::cout << "(" << item.first << ") ";
        }
        std::cout << std::endl;
    }
};

/**
 * @class MinHeap
 * @brief A minimum heap implementation based on HeapBase.
 *
 * Maintains a min-heap where the smallest key is always at the root.
 */
template <typename T>
class MinHeap : public HeapBase<T> {
protected:
    void heapifyUp(int index) override {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (this->data[index].first >= this->data[parent].first) break;

            std::swap(this->data[index], this->data[parent]);
            index = parent;
        }
    }

    void heapifyDown(int index) override {
        int size = this->data.size();
        while (true) {
            int left = 2 * index + 1;
            int right = 2 * index + 2;
            int smallest = index;

            if (left < size && (this->data[left].first < this->data[smallest].first)) {
                smallest = left;
            }
            if (right < size && (this->data[right].first < this->data[smallest].first)) {
                smallest = right;
            }
            if (smallest == index) break;

            std::swap(this->data[index], this->data[smallest]);
            index = smallest;
        }
    }
};

/**
 * @class MaxHeap
 * @brief A maximum heap implementation based on HeapBase.
 *
 * Maintains a max-heap where the largest key is always at the root.
 */
template <typename T>
class MaxHeap : public HeapBase<T> {
protected:
    void heapifyUp(int index) override {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (this->data[index].first > this->data[parent].first) {
                std::swap(this->data[index], this->data[parent]);
                index = parent;
            } else {
                break;
            }
        }
    }

    void heapifyDown(int index) override {
        int size = this->data.size();
        while (2 * index + 1 < size) {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            int largest = index;

            if (leftChild < (size && this->data[leftChild].first > this->data[largest].first)) {
                largest = leftChild;
            }
            if (rightChild < (size && this->data[rightChild].first > this->data[largest].first)) {
                largest = rightChild;
            }
            if (largest != index) {
                std::swap(this->data[index], this->data[largest]);
                index = largest;
            } else {
                break;
            }
        }
    }
};

/**
 * @class MinMaxHeap
 * @brief A min-max heap implementation based on HeapBase.
 */
 template <typename T>
 class MinMaxHeap : public HeapBase<T> {
 protected:
    /** @brief Check if is in Min level. */
    bool isMinLevel(int index) {
        return (static_cast<int>(log2(index + 1)) % 2 == 0);
    }
 
    void heapifyUp(int index) override {
        if (index == 0) return;
 
        int parent = (index - 1) / 2;
        if (isMinLevel(index)) { // In Min level.
            if (this->data[index].first > this->data[parent].first) { // Need to swap to Max level.
                std::swap(this->data[index], this->data[parent]);
                heapifyUpMax(parent);
            } else {
                heapifyUpMin(index);
            }
        } else { // In Max level.
            if (this->data[index].first < this->data[parent].first) { // Need to swap to Min level.
                std::swap(this->data[index], this->data[parent]);
                heapifyUpMin(parent);
            } else {
                heapifyUpMax(index);
            }
        }
    }
 
    void heapifyUpMin(int index) {
        if (index <= 2) return; // No grandparent.
        int grandparent = (((index - 1) / 2) - 1) / 2;
        if (this->data[index].first < this->data[grandparent].first) {
            std::swap(this->data[index], this->data[grandparent]);
            heapifyUpMin(grandparent);
        }
     }

    void heapifyUpMax(int index) {
        if (index <= 2) return; // No grandparent.
        int grandparent = (((index - 1) / 2) - 1) / 2;
        if (this->data[index].first > this->data[grandparent].first) {
            std::swap(this->data[index], this->data[grandparent]);
            heapifyUpMax(grandparent);
        }
    }

    void heapifyDown(int index) override {
        if (isMinLevel(index)) {
            heapifyDownMin(index);
        } else {
            heapifyDownMax(index);
        }
    }
 
    void heapifyDownMin(int index) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        int smallest = index;
 
        if (left < this->data.size() && this->data[left].first < this->data[smallest].first) {
            smallest = left;
        }
        if (right < this->data.size() && this->data[right].first < this->data[smallest].first) {
             smallest = right;
        }
 
        if (smallest != index) {
            std::swap(this->data[index], this->data[smallest]);
            heapifyDownMin(smallest);
        }
    }

    void heapifyDownMax(int index) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        int largest = index;
 
        if (left < this->data.size() && this->data[left].first > this->data[largest].first) {
            largest = left;
        }
        if (right < this->data.size() && this->data[right].first > this->data[largest].first) {
            largest = right;
        }
 
        if (largest != index) {
            std::swap(this->data[index], this->data[largest]);
            heapifyDownMax(largest);
        }
    }
 
public:
    T getMin() {
        if (this->data.empty()) throw std::runtime_error("Heap is empty!");

        return this->data[0];
    }
 
    T getMax() {
        if (this->data.empty()) throw std::runtime_error("Heap is empty!");
        if (this->data.size() == 1) return this->data[0];
        if (this->data.size() == 2) return this->data[1];

        return std::max(this->data[1], this->data[2]); // Max value is in second node or third node.
    }
 
    std::pair<int, T> popMin() {
        if (this->data.empty()) throw std::runtime_error("Heap is empty!");
        std::pair<int, T> popData = this->data[0];
        this->data[0] = this->data.back();
        this->data.pop_back();
        heapifyDown(0);

        return popData;
    }
 
    std::pair<int, T> popMax() {
        if (this->data.empty()) throw std::runtime_error("Heap is empty!");
        if (this->data.size() == 1) {
            std::pair<int, T> popData = this->data[0];
            this->data.pop_back();
            return popData;
        }
        int maxIndex = (this->data.size() == 2) ? 1 : (this->data[1].first > this->data[2].first ? 1 : 2);
        std::pair<int, T> popData = this->data[0];
        this->data[maxIndex] = this->data.back();
        this->data.pop_back();
        heapifyDown(maxIndex);

        return popData;
    }
};

/**
 * @class Deap
 * @brief A dual heap structure containing both a min-heap and a max-heap.
 */
template <typename T>
class Deap {
private:
    MinHeap<T> minHeap;
    MaxHeap<T> maxHeap;

public:
    bool isEmpty() const {
        return minHeap.isEmpty() && maxHeap.isEmpty();
    }

    void clear() {
        minHeap.clear();
        maxHeap.clear();
    }

    const T& topMin() const { return minHeap.top(); }

    const T& topMax() const { return maxHeap.top(); }

    const std::pair<int, T>& bottom() const {
        if (minHeap.isEmpty() && maxHeap.isEmpty()) {
            throw std::runtime_error("Heap is empty!");
        } else if (minHeap.height() > maxHeap.height()) {
            return minHeap.bottom();
        } else {
            return maxHeap.bottom();
        }
    }

    const std::pair<int, T>& leftmostBottom() const {
        if (minHeap.isEmpty()) {
            throw std::runtime_error("Heap is empty!");
        }

        return minHeap.leftmostBottom();
    }

    void push(int key, const T& data) {
        #ifdef DEBUG
            std::cout << "Test: " << minHeap.height(1) << " and " << maxHeap.height() << " ; " << key << std::endl;
        #endif
        if (minHeap.size() == maxHeap.size() ||
            ((minHeap.height(1) - maxHeap.height()) == 1 && maxHeap.isFullBinaryTree(maxHeap.size()))) { 
            // Insert min heap.
            
            if (minHeap.isEmpty()) {
                minHeap.push(key, data);
            } else {
                int parent = (minHeap.size() - 1) / 2;
                if (key < maxHeap.getAt(parent).first) {
                    minHeap.push(key, data);
                } else {
                    std::pair<int, T> replacement = maxHeap.replaceAt(parent, key, data);
                    minHeap.push(replacement.first, replacement.second);
                }
            }
        } else { // Insert max heap.
            int replaceIndex = maxHeap.size();
            if (key > minHeap.getAt(replaceIndex).first) {
                maxHeap.push(key, data);
            } else {
                std::pair<int, T> replacement = minHeap.replaceAt(replaceIndex, key, data);
                maxHeap.push(replacement.first, replacement.second);
            }
        }
    }

    std::pair<int, T> popMin() {
        if (minHeap.isEmpty()) throw std::runtime_error("Min heap is empty!");
        std::pair<int, T> popData = minHeap.pop();

        if (minHeap.size() < maxHeap.size()) {
            minHeap.push(maxHeap.top().first, maxHeap.top().second);
            maxHeap.pop();
        }

        return popData;
    }

    std::pair<int, T> popMax() {
        if (maxHeap.isEmpty()) throw std::runtime_error("Max heap is empty!");
        std::pair<int, T> popData = maxHeap.pop();

        if (maxHeap.size() < minHeap.size() - 1) {
            maxHeap.push(minHeap.top().first, minHeap.top().second);
            minHeap.pop();
        }

        return popData;
    }

    void printHeaps() const {
        std::cout << "MinHeap: ";
        minHeap.printHeap();

        std::cout << "MaxHeap: ";
        maxHeap.printHeap();
    }
};

 /**
 * @class UniversityDepartment
 *
 * @brief Represents graduate information.
 */
class UniversityDepartment {
    private:
    int order;
    int schoolCode;
    std::string schoolName;
    int deptCode;
    std::string deptName;
    std::string dayNight;
    std::string degree;
    int studentNum;
    int teacherNum;
    int graduateNum;
    std::string city;
    std::string systemType;
    
    public:
        // Constructor
        UniversityDepartment(int order, int schoolCode, const std::string& schoolName, int deptCode, const std::string& deptName,
                             const std::string& dayNight, const std::string& degree, int studentNum, int teacherNum, 
                             int graduateNum, const std::string& city, const std::string& systemType)
            : order(order), schoolCode(schoolCode), schoolName(schoolName), deptCode(deptCode), deptName(deptName), 
              dayNight(dayNight), degree(degree), studentNum(studentNum), teacherNum(teacherNum), 
              graduateNum(graduateNum), city(city), systemType(systemType) {}
    
        // Getters
        int getOrder() const { return order; }
        int getSchoolCode() const { return schoolCode; }
        std::string getSchoolName() const { return schoolName; }
        int getDeptCode() const { return deptCode; }
        std::string getDeptName() const { return deptName; }
        std::string getDayNight() const { return dayNight; }
        std::string getDegree() const { return degree; }
        int getStudentNum() const { return studentNum; }
        int getTeacherNum() const { return teacherNum; }
        int getGraduateNum() const { return graduateNum; }
        std::string getCity() const { return city; }
        std::string getSystemType() const { return systemType; }
    
        // Setters
        void setOrder(int orderVal) { order = orderVal; }
        void setSchoolCode(int schoolCodeVal) { schoolCode = schoolCodeVal; }
        void setSchoolName(const std::string& schoolNameVal) { schoolName = schoolNameVal; }
        void setDeptCode(int deptCodeVal) { deptCode = deptCodeVal; }
        void setDeptName(const std::string& deptNameVal) { deptName = deptNameVal; }
        void setDayNight(const std::string& dayNightVal) { dayNight = dayNightVal; }
        void setDegree(const std::string& degreeVal) { degree = degreeVal; }
        void setStudentNum(int studentNumVal) { studentNum = studentNumVal; }
        void setTeacherNum(int teacherNumVal) { teacherNum = teacherNumVal; }
        void setGraduateNum(int graduateNumVal) { graduateNum = graduateNumVal; }
        void setCity(const std::string& cityVal) { city = cityVal; }
        void setSystemType(const std::string& systemTypeVal) { systemType = systemTypeVal; }
};

/**
 * @brief Removes all non-numeric characters from the given string.
 * 
 * @param str The input string to be cleaned.
 * @return A string containing only the numeric characters from the input string.
 */
std::string removeNonNumeric(const std::string& str) {
    std::string cleanedStr;
    for (const char& c : str) {
        if (std::isdigit(c)) {
            cleanedStr += c;
        }
    }
    return cleanedStr;
}

/**
 * @brief Reads graduate data from a tab-separated file.
 *
 * This function reads data from the specified input file and populates the given vector with graduate objects.
 * The input file is expected to be in TSV (Tab-Separated Values) format, where each row contains
 * the details of a graduate.
 *
 * @param inputFileName The name of the input file to read.
 * @param graduateInfoList A reference to a vector where graduate objects will be stored.
 * @return `true` if the file is successfully read, `false` if the file cannot be opened.
 *
 * @details
 * - The first line of the file is assumed to be a header and is skipped.
 * - If a row is invalid, it will be ignored.
 */
 bool ReadFile(const std::string& inputFileName, std::vector<UniversityDepartment>& graduateInfoList){
    std::ifstream inputFile(inputFileName);

    // Make sure file exist.
    if (!inputFile.is_open()) {
        std::cout << std::endl;
        std::cout << "### " << inputFileName << " does not exist! ###" << std::endl;
        std::cout << std::endl;

        // File can't be opened.
        return false;
    }

    // Skip head of three lines.
    std::string headerLine;
    for (int i = 0; i < 3; ++i) {
        std::getline(inputFile, headerLine);
    }
    
    // Get data.
    std::string line;
    int order = 0;
    while (std::getline(inputFile, line)) {
        std::istringstream line_stream(line);
        std::vector<std::string> graduateInformationParam;
        std::string token;

        // Separate the data by "Tab".
        while (std::getline(line_stream, token, '\t')) {
            graduateInformationParam.push_back(token);
        }

        // TODO: When the format does not meet the need(TSV) to throw an exception.
        // TODO: If the number of graduateInformationParam is not equal to the expected value, 
        //       it should be skipped instead of allowing the line that failed to parse to enter.
        // Make sure have data.
        if (graduateInformationParam.size() != 0) {
            ++order;
            int schoolCode = stoi(removeNonNumeric(graduateInformationParam[0]));
            std::string schoolName = graduateInformationParam[1];
            int deptCode = stoi(removeNonNumeric(graduateInformationParam[2]));
            std::string deptName = graduateInformationParam[3];
            std::string dayNight = graduateInformationParam[4];
            std::string degree = graduateInformationParam[5];
            int studentNum = stoi(removeNonNumeric(graduateInformationParam[6]));
            int teacherNum = stoi(removeNonNumeric(graduateInformationParam[7]));
            int graduateNum = stoi(removeNonNumeric(graduateInformationParam[8]));
            std::string city = graduateInformationParam[9];
            std::string systemType = graduateInformationParam[10];

            // Create and store the graduateInformation.
            UniversityDepartment graduateInfo(order, schoolCode, schoolName, deptCode, deptName, dayNight, degree, studentNum, teacherNum, graduateNum, city, systemType);
            graduateInfoList.push_back(graduateInfo);
        }
    }

    inputFile.close();

    // Opened file success.
    return true;
}

/**
 * @brief Sorts a vector of graduate objects by their school code values in descending order.
 *
 * If two graduate have the same school code, they are sorted by their index in ascending order.
 *
 * @param graduateInfoList A reference to a vector of graduate objects to be sorted.
 *
 * @details
 * - This function uses `std::sort` with a custom comparison lambda function.
 * - Sorting is performed in-place, modifying the original vector.
 * - Lower school code values appear first.
 */
 void sortGraduateInfoList(std::vector<UniversityDepartment>& graduateInfoList) {
    std::sort(graduateInfoList.begin(), graduateInfoList.end(),
        [](const UniversityDepartment& a, const UniversityDepartment& b) {
            return a.getSchoolCode() < b.getSchoolCode();
        }
    );
}

/**
 * @brief Prints the saved graduate data in a formatted table.
 * 
 * @param graduateInfoList The list of university department data to print.
 */
void printSaveData(std::vector<UniversityDepartment>& graduateInfoList) {
    for (int i = 0; i < graduateInfoList.size(); ++i) {
        UniversityDepartment graduateInfo = graduateInfoList[i];

        std::cout << "[" << i + 1 << "]" << "\t" << std::setw(5) << std::right
                  << graduateInfo.getSchoolName() << "\t" << std::setw(4) << std::right
                  << graduateInfo.getDeptName() << "\t" << std::setw(4) << std::right
                  << graduateInfo.getDayNight() << "\t" << std::setw(4) << std::right
                  << graduateInfo.getDegree() << "\t" << std::setw(2) << std::right
                  << graduateInfo.getStudentNum() << "\t" << std::setw(1) << std::right
                  << graduateInfo.getTeacherNum() << "\t" << std::setw(2) << std::right
                  << graduateInfo.getGraduateNum()
                  << std::endl;
    }
}

/** @brief Processes graduate data using a min heap. */
void Task1(MinHeap<UniversityDepartment> graduateInfo) {
    if (!graduateInfo.isEmpty()) graduateInfo.clear();

    std::vector<UniversityDepartment> graduateInfoList;

    // Continue to ask the user to enter the file number until the user chooses to exit.
    while (true) {
        std::string inputFileName = "";

        std::cout << "Input a file number ([0] Quit): ";
        std::cin >> inputFileName;

        if ("0" == inputFileName) {
            return; // Exit if the user enters 0.
        } else {
            inputFileName = "input" + inputFileName + ".txt";

            // Try to open the file.
            // If fail, enter the file name again.
            if (!ReadFile(inputFileName, graduateInfoList)) continue;

            // Check if the file contains any data.
            if(!graduateInfoList.empty()) {
                break; // Read data success, jump out the loop.
            } else {
                std::cout << std::endl;
                std::cout << "### Get nothing from the file "<< inputFileName <<" ! ###" << std::endl;
                return;
            }
        }
    }

    // Build Min heap.
    for (const UniversityDepartment& info : graduateInfoList) {
        graduateInfo.push(info.getGraduateNum(), info);
    }

    #ifdef DEBUG
        graduateInfo.printHeap();
    #endif

    // Output information.
    std::cout << "<min heap>" << std::endl;
    UniversityDepartment root = graduateInfo.top().second;
    std::cout << "root: [" << root.getOrder() << "] " << root.getGraduateNum() << std::endl;
    UniversityDepartment bottom = graduateInfo.bottom().second;
    std::cout << "bottom: [" << bottom.getOrder() << "] " << bottom.getGraduateNum() << std::endl;
    UniversityDepartment leftmostBottom = graduateInfo.leftmostBottom().second;
    std::cout << "leftmost bottom: [" << leftmostBottom.getOrder() << "] " << leftmostBottom.getGraduateNum() << std::endl;

    #ifdef DEBUG
        MaxHeap<UniversityDepartment> maxHeap;
        
        for (const UniversityDepartment& info : graduateInfoList) {
            maxHeap.push(info.getGraduateNum(), info);
        }

        std::cout << "FOR TEST" << std::endl;
        maxHeap.printHeap();

        std::cout << "<max heap>" << std::endl;
        root = maxHeap.top().second;
        std::cout << "root: [" << root.getOrder() << "] " << root.getGraduateNum() << std::endl;
        bottom = maxHeap.bottom().second;
        std::cout << "bottom: [" << bottom.getOrder() << "] " << bottom.getGraduateNum() << std::endl;
        leftmostBottom = maxHeap.leftmostBottom().second;
        std::cout << "leftmost bottom: [" << leftmostBottom.getOrder() << "] " << leftmostBottom.getGraduateNum() << std::endl;
    #endif
}

/** @brief Processes graduate data using a deap. */
void Task2(Deap<UniversityDepartment> graduateInfo) {
    if (!graduateInfo.isEmpty()) graduateInfo.clear();

    std::vector<UniversityDepartment> graduateInfoList;

    // Continue to ask the user to enter the file number until the user chooses to exit.
    while (true) {
        std::string inputFileName = "";

        std::cout << "Input a file number ([0] Quit): ";
        std::cin >> inputFileName;

        if ("0" == inputFileName) {
            return; // Exit if the user enters 0.
        } else {
            inputFileName = "input" + inputFileName + ".txt";

            // Try to open the file.
            // If fail, enter the file name again.
            if (!ReadFile(inputFileName, graduateInfoList)) continue;

            // Check if the file contains any data.
            if(!graduateInfoList.empty()) {
                break; // Read data success, jump out the loop.
            } else {
                std::cout << std::endl;
                std::cout << "### Get nothing from the file "<< inputFileName <<" ! ###" << std::endl;
                return;
            }
        }
    }

    // Build Deap.
    for (const UniversityDepartment& info : graduateInfoList) {
        graduateInfo.push(info.getStudentNum(), info);
    }

    #ifdef DEBUG
    graduateInfo.printHeaps();
    #endif

    // Output information.
    std::cout << "<DEAP>" << std::endl;
    UniversityDepartment bottom = graduateInfo.bottom().second;
    std::cout << "bottom: [" << bottom.getOrder() << "] " << bottom.getStudentNum() << std::endl;
    UniversityDepartment leftmostBottom = graduateInfo.leftmostBottom().second;
    std::cout << "leftmost bottom: [" << leftmostBottom.getOrder() << "] " << leftmostBottom.getStudentNum() << std::endl;
}

/** @brief Processes graduate data using a deap. */
void Task3(MinMaxHeap<UniversityDepartment> graduateInfo) {
    if (!graduateInfo.isEmpty()) graduateInfo.clear();

    std::vector<UniversityDepartment> graduateInfoList;

    // Continue to ask the user to enter the file number until the user chooses to exit.
    while (true) {
        std::string inputFileName = "";

        std::cout << "Input a file number ([0] Quit): ";
        std::cin >> inputFileName;

        if ("0" == inputFileName) {
            return; // Exit if the user enters 0.
        } else {
            inputFileName = "input" + inputFileName + ".txt";

            // Try to open the file.
            // If fail, enter the file name again.
            if (!ReadFile(inputFileName, graduateInfoList)) continue;

            // Check if the file contains any data.
            if(!graduateInfoList.empty()) {
                break; // Read data success, jump out the loop.
            } else {
                std::cout << std::endl;
                std::cout << "### Get nothing from the file "<< inputFileName <<" ! ###" << std::endl;
                return;
            }
        }
    }

    // Build Min heap.
    for (const UniversityDepartment& info : graduateInfoList) {
        graduateInfo.push(info.getGraduateNum(), info);
    }

    #ifdef DEBUG
        graduateInfo.printHeap();
    #endif

    // Output information.
    std::cout << "<min-max heap>" << std::endl;
    UniversityDepartment root = graduateInfo.top().second;
    std::cout << "root: [" << root.getOrder() << "] " << root.getGraduateNum() << std::endl;
    UniversityDepartment bottom = graduateInfo.bottom().second;
    std::cout << "bottom: [" << bottom.getOrder() << "] " << bottom.getGraduateNum() << std::endl;
    UniversityDepartment leftmostBottom = graduateInfo.leftmostBottom().second;
    std::cout << "leftmost bottom: [" << leftmostBottom.getOrder() << "] " << leftmostBottom.getGraduateNum() << std::endl;
}

int main() {
    int select_command = 0;
    int select_lower_bound = 0;
    int select_upper_bound = 3;
    MinHeap<UniversityDepartment> graduateInfoMinHeap;
    Deap<UniversityDepartment> graduateInfoDeap;
    MinMaxHeap<UniversityDepartment> graduateInfoMinMaxHeap;

    do {
        while (true) {
            std::cout <<
                "**** Heap Construction *****\n"
                "* 0. QUIT                  *\n"
                "* 1. Build a min heap    *\n"
                "* 2. Build a DEAP          *\n"
                "* 3. Build a min-max heap  *\n"
                "****************************\n"
                "Input a choice(0, 1, 2, 3): ";

            std::cin >> select_command;

            if (!std::cin.fail()) {
                break;
            } else {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                std::cout << std::endl;
                std::cout << "Command does not exist!" << std::endl;
                std::cout << std::endl;  
            }
        }

        switch (select_command) {
        case 0:
            break;
        case 1:
            std::cout << std::endl;
            Task1(graduateInfoMinHeap);
            std::cout << std::endl;
            break;
        case 2:
            std::cout << std::endl;
            Task2(graduateInfoDeap);
            std::cout << std::endl;
            break;
        case 3:
            std::cout << std::endl;
            Task3(graduateInfoMinMaxHeap);
            std::cout << std::endl;
            break;
        default:
            std::cout << std::endl;
            std::cout << "Command does not exist!" << std::endl;
            std::cout << std::endl;
        }
    } while (select_command != 0);

    return 0;
}