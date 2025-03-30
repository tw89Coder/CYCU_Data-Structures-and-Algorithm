/** 
 * @file DS2ex02_27_10927262.cpp
 * @brief A program that uses 2-3 tree and avl tree to manage graduate data.
 * @version 1.0.1
 *
 * @details
 * This program reads and processes graduate data using a tree structure.
 *
 * @author 
 * - Group 27
 * - 10927262 呂易鴻
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 節點類別 (BTreeNode)
// 每個節點包含以下成員：
// 1. `keys` - 存放節點內所有的鍵值，保持排序狀態
// 2. `children` - 指向子節點的指標，表示樹的分支
// 3. `data` - 每個鍵值對應的資料，類型為 T
template <typename T>
class BTreeNode {
public:
    std::vector<std::string> keys; // 鍵值的集合，表示該節點內的數據 (修改為 string)
    std::vector<std::vector<T>> data; // 每個鍵值對應的資料集合
    std::vector<BTreeNode*> children; // 子節點的指標列表，用來指向下一層的節點
    int height; // 節點的高度

    // 預設建構子
    BTreeNode() : height(1) {} // 默認高度為 1 (葉子節點的高度)

    // 解構子，負責釋放子節點記憶體
    ~BTreeNode() {
        for (auto child : children) {
            delete child;
        }
    }
};

// B 樹類別 (BTree)
// 此類別包含 B 樹的主要邏輯，包括插入與節點分裂
template <typename T>
class BTree {
public:
    // 建構子
    // 接收樹的階數 `order` 並初始化根節點
    // 使用者可以選擇排序方式：字典排序 ("lexicographic") 或數字排序 ("numeric")
    explicit BTree(int order, const std::string& mode = "B-tree", const std::string& sort_type = "lexicographic")
        : order_(order), mode_(mode), sort_type_(sort_type), root_(new BTreeNode<T>) {
        maxKeys_ = getMaxKeys();
    }

    ~BTree() {
        delete root_;
    }

    // 重新初始化整棵樹
    void clear() {
        delete root_; // 先刪除原本的樹
        root_ = new BTreeNode<T>(); // 創建一個新的空樹
    }

    // 插入函式
    // 將新的鍵值插入到 B 樹中
    void insert(const std::string& key, const T& data) {
        insertInternal(root_, key, data, maxKeys_);
    
        // 在插入完成後檢查根節點是否超出限制
        if (root_->keys.size() > maxKeys_) {
            BTreeNode<T>* temp = root_;
            root_ = new BTreeNode<T>();
            root_->children.push_back(temp);
            split(root_, 0, maxKeys_); // 這裡才執行分裂
        }
    }    
    

    BTreeNode<T>* getRoot() const {
        return root_;
    }

    // 獲取指定節點的所有資料
    std::vector<T> getRootData() const {
        std::vector<T> nodeData;
        // 遍歷節點的資料
        for (const auto& dataVec : root_->data) {
            if (!dataVec.empty()) {
                // 將 dataVec 中的每個元素加入 nodeData
                for (const auto& data : dataVec) {
                    nodeData.push_back(data);  // 正確地插入每個元素
                }
            }
        }

        return nodeData;
    }

    // 取得 B 樹的高度
    int getHeight() const {
        return root_->height; // 直接返回根節點的高度
    }

    // 印出 B 樹的結構 (層次顯示)
    void printBtree() const {
        // 使用佇列實現層次遍歷
        std::vector<std::pair<BTreeNode<T>*, int>> queue;
        queue.emplace_back(root_, 1); // 從根節點（高度 1）開始
    
        while (!queue.empty()) {
            BTreeNode<T>* node = queue.front().first;  // 目前處理的節點
            int level = queue.front().second;         // 該節點的高度
            queue.erase(queue.begin());
    
            // 在每個節點前顯示它的高度 level
            std::cout << "Level " << level << ": { ";
    
            // 如果節點有鍵，遍歷顯示
            if (!node->keys.empty()) {
                for (size_t i = 0; i < node->keys.size(); ++i) {
                    std::cout << "[" << node->keys[i] << ": (";
                    // 如果節點有資料，遍歷顯示資料
                    if (!node->data[i].empty()) {
                        for (size_t j = 0; j < node->data[i].size(); ++j) {
                            std::cout << node->data[i][j];
                            if (j != node->data[i].size() - 1) {
                                std::cout << ", ";
                            }
                        }
                    }
                    std::cout << ")]";
                    if (i != node->keys.size() - 1) {
                        std::cout << ", "; // 鍵值之間的逗號分隔
                    }
                }
            }
            std::cout << " }" << std::endl;
    
            // 將子節點加入佇列，並更新高度 (level + 1)
            for (BTreeNode<T>* child : node->children) {
                queue.emplace_back(child, level + 1);
            }
        }
    }    

private:
    int order_; // 樹的階數，決定每個節點的最大鍵值數量
    int maxKeys_; // 根據模式取得最大鍵值數量
    std::string mode_;
    std::string sort_type_; // 排序類型 ("lexicographic" 或 "numeric")
    BTreeNode<T>* root_; // 樹的根節點指標

    // 根據模式返回節點的最大鍵值數量
    int getMaxKeys() const {
        if (mode_ == "2-3 tree") {
            return 2; // 2-3 tree 的節點最多 2 個鍵值
        }
        return 2 * order_ - 1; // 一般 B-tree
    }

    // 插入內部函式
    // 遞迴將鍵值插入到適當的節點位置
    // 插入內部函式
    // 遞迴將鍵值插入到適當的節點位置
    void insertInternal(BTreeNode<T>* node, const std::string& key, const T& data, int maxKeys_) {
        if (node->children.empty()) {
            // 葉節點
            int index = getChildIndex(node, key);
            if (index < node->keys.size() && node->keys[index] == key) {
                node->data[index].push_back(data); // 如果 key 存在，直接追加 data
            } else {
                node->keys.insert(node->keys.begin() + index, key);
                node->data.insert(node->data.begin() + index, std::vector<T>{data});
            }
        } else {
            // 非葉節點
            int index = getChildIndex(node, key);
            if (index < node->keys.size() && node->keys[index] == key) {
                // 如果非葉節點中已存在相同的 key，直接追加 data
                node->data[index].push_back(data);
            } else {
                // 遞迴插入至子節點
                insertInternal(node->children[index], key, data, maxKeys_);
    
                // 如果子節點分裂，檢查是否需要進一步處理
                if (node->children[index]->keys.size() > maxKeys_) {
                    split(node, index, maxKeys_);
                    if (compare(node->keys[index], key) < 0) {
                        ++index; // 更新索引，確保插入到正確的子節點中
                    }
                }
            }
        }
    
        // 更新節點高度
        if (mode_ == "2-3 tree") {
            node->height = 1 + (node->children.empty() ? 0 : node->children[0]->height);
        } else {
            node->height = std::max(node->height, (node->children.empty() ? 1 : node->children[0]->height + 1));
        }
    }    

    // 分裂節點函式
    // 將已滿的節點分裂為兩個節點，並將中間鍵值提升到父節點
    // 分裂節點函式
    void split(BTreeNode<T>* parent, int index, int maxKeys_) {
        BTreeNode<T>* nodeToSplit = parent->children[index];
        BTreeNode<T>* newNode = new BTreeNode<T>();
    
        int mid = maxKeys_ / 2;  // 對於 2-3 Tree, maxKeys_ 是 2，所以 mid = 1
    
        // 確保在 2-3 Tree 模式中，maxKeys_ 必須是 2
        if (mode_ == "2-3 tree" && maxKeys_ != 2) {
            delete newNode;
            std::cerr << "錯誤：在 2-3 tree 模式下，maxKeys_ 必須是 2！" << std::endl;
            return;
        }
    
        // 將中間鍵值提升到父節點
        const std::string& midKey = nodeToSplit->keys[mid];
        auto& midData = nodeToSplit->data[mid];
    
        parent->keys.insert(parent->keys.begin() + index, midKey);
        parent->data.insert(parent->data.begin() + index, midData);
    
        // 將右半部分鍵值與資料分配給新節點
        newNode->keys.assign(nodeToSplit->keys.begin() + mid + 1, nodeToSplit->keys.end());
        newNode->data.assign(nodeToSplit->data.begin() + mid + 1, nodeToSplit->data.end());
    
        // 移除已經移動的鍵值與資料
        nodeToSplit->keys.resize(mid);
        nodeToSplit->data.resize(mid);
    
        // 如果不是葉節點，處理子節點
        if (!nodeToSplit->children.empty()) {
            newNode->children.assign(nodeToSplit->children.begin() + mid + 1, nodeToSplit->children.end());
            nodeToSplit->children.resize(mid + 1);
        }
    
        // 將新節點插入父節點的子節點中
        parent->children.insert(parent->children.begin() + index + 1, newNode);
    
        // 更新父節點的高度
        parent->height = std::max(parent->children[index]->height, parent->children[index + 1]->height) + 1;
    }     
    


    // 找出鍵值應插入的子節點索引
    int getChildIndex(BTreeNode<T>* node, const std::string& key) const {
        int i = 0;
        // 使用比較函式比較鍵值
        while (i < node->keys.size() && compare(node->keys[i], key) < 0) {
            if (compare(node->keys[i], key) == 0) return i; // 返回相同鍵值的索引
            ++i;
        }
    
        return i;  // 返回插入的位置
    }    

    // 比較函式，根據排序方式選擇字典排序或數字排序
    int compare(const std::string& a, const std::string& b) const {
        if (sort_type_ == "numeric") {
            // 嘗試將字串轉為數字來進行比較
            try {
                int num_a = std::stoi(a);
                int num_b = std::stoi(b);
                return num_a - num_b;
            } catch (const std::invalid_argument&) {
                // 如果轉換失敗，回傳 0，避免出錯
                return a.compare(b);
            }
        } else {
            // 默認使用字典排序
            // a < b return -1
            return a.compare(b);
        }
    }
};

template <typename T>
class AVLTreeNode {
public:
    std::string key;  // 鍵值
    std::vector<T> data;  // 對應資料
    AVLTreeNode* left;  // 左子節點
    AVLTreeNode* right;  // 右子節點
    int height;  // 樹的高度

    // 預設建構子
    AVLTreeNode(const std::string& key, const T& data)
        : key(key), data({data}), left(nullptr), right(nullptr), height(1) {}

    // 解構子
    ~AVLTreeNode() {
        left = nullptr;
        right = nullptr;
    }
};

template <typename T>
class AVLTree {
public:
    explicit AVLTree(const std::string& sort_type = "lexicographic")
        : sort_type_(sort_type), root_(nullptr) {}

    ~AVLTree() {
        clear();
    }

    void clear() {
        clearInternal(root_);
        root_ = nullptr;
    }

    void clearInternal(AVLTreeNode<T>* node) {
        if (node == nullptr) return;

        clearInternal(node->left);
        clearInternal(node->right);
        delete node;  // Only delete the node once
    }

    bool isEmpty() const {
        return root_ == nullptr;
    }

    // 插入函式
    void insert(const std::string& key, const T& data) {
        root_ = insertInternal(root_, key, data);
    }

    // 取得樹根
    AVLTreeNode<T>* getRoot() const {
        return root_;
    }

    std::vector<T> getRootData() const {
        if (root_ == nullptr) {
            throw std::runtime_error("Tree root is null");
        }
        return root_->data;
    }

    int getHeight() const {
        return isEmpty() ? 0 : root_->height;
    }

    // 印出樹的結構
    void printTree() const {
        printTreeInternal(root_, 0);
    }

private:
    std::string sort_type_;  // 排序類型 ("lexicographic" 或 "numeric")
    AVLTreeNode<T>* root_;  // 樹的根節點

    // 插入內部函式，保持樹的平衡
    AVLTreeNode<T>* insertInternal(AVLTreeNode<T>* node, const std::string& key, const T& data) {
        if (node == nullptr) {
            return new AVLTreeNode<T>(key, data);  // 創建新節點
        }

        // 標準的二叉搜尋樹插入
        if (compare(key, node->key) < 0) {
            node->left = insertInternal(node->left, key, data);
        } else if (compare(key, node->key) > 0) {
            node->right = insertInternal(node->right, key, data);
        } else {
            // 如果相同鍵值存在，直接新增資料
            node->data.push_back(data);
            return node;
        }

        // 更新節點高度
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));

        // 計算平衡因子並進行旋轉
        int balance = getBalance(node);

        // 左左情況
        if (balance > 1 && compare(key, node->left->key) < 0) {
            return rightRotate(node);
        }

        // 右右情況
        if (balance < -1 && compare(key, node->right->key) > 0) {
            return leftRotate(node);
        }

        // 左右情況
        if (balance > 1 && compare(key, node->left->key) > 0) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }

        // 右左情況
        if (balance < -1 && compare(key, node->right->key) < 0) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    // 左旋轉操作
    AVLTreeNode<T>* leftRotate(AVLTreeNode<T>* x) {
        AVLTreeNode<T>* y = x->right;
        AVLTreeNode<T>* T2 = y->left;

        // 進行旋轉
        y->left = x;
        x->right = T2;

        // 更新高度
        x->height = 1 + std::max(getHeight(x->left), getHeight(x->right));
        y->height = 1 + std::max(getHeight(y->left), getHeight(y->right));

        return y;  // 返回新根
    }

    // 右旋轉操作
    AVLTreeNode<T>* rightRotate(AVLTreeNode<T>* y) {
        AVLTreeNode<T>* x = y->left;
        AVLTreeNode<T>* T2 = x->right;

        // 進行旋轉
        x->right = y;
        y->left = T2;

        // 更新高度
        y->height = 1 + std::max(getHeight(y->left), getHeight(y->right));
        x->height = 1 + std::max(getHeight(x->left), getHeight(x->right));

        return x;  // 返回新根
    }

    // 取得節點高度
    int getHeight(AVLTreeNode<T>* node) const {
        if (node == nullptr) {
            return 0;
        }
        return node->height;
    }

    // 取得節點的平衡因子
    int getBalance(AVLTreeNode<T>* node) const {
        if (node == nullptr) {
            return 0;
        }
        return getHeight(node->left) - getHeight(node->right);
    }

    // 比較函式，根據排序方式選擇字典排序或數字排序
    int compare(const std::string& a, const std::string& b) const {
        if (sort_type_ == "numeric") {
            // 嘗試將字串轉為數字來進行比較
            try {
                int num_a = std::stoi(a);
                int num_b = std::stoi(b);
                return num_a - num_b;
            } catch (const std::invalid_argument&) {
                // 如果轉換失敗，回傳 0，避免出錯
                return a.compare(b);
            }
        } else {
            // 默認使用字典排序
            // a < b return -1
            return a.compare(b);
        }
    }

    // 層次遍歷印出樹結構
    void printTreeInternal(AVLTreeNode<T>* node, int level) const {
        if (node == nullptr) {
            return;
        }

        // 印出當前層次的節點
        std::cout << "Level " << level << ": { " << node->key << ": [";
        for (size_t i = 0; i < node->data.size(); ++i) {
            std::cout << node->data[i];
            if (i != node->data.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "] }" << std::endl;

        // 遞迴印出左右子樹
        printTreeInternal(node->left, level + 1);
        printTreeInternal(node->right, level + 1);
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
void printSaveData(const std::vector<UniversityDepartment>& graduateInfoList, const std::vector<int>& order) {
    for (int i = 0; i < order.size(); ++i) {
        int index = order[i];
        UniversityDepartment graduateInfo = graduateInfoList[index - 1];

        std::cout << i + 1 << ": "
                  << "[" << graduateInfo.getOrder() << "]" << " "
                  << graduateInfo.getSchoolName() << ", "
                  << graduateInfo.getDeptName() << ", "
                  << graduateInfo.getDayNight() << ", "
                  << graduateInfo.getDegree() << ", "
                  << graduateInfo.getStudentNum()
                  << std::endl;
                  
    }
    
    std::cout << std::endl;
}

/** @brief Processes graduate data using a min heap. */
void Task1(std::vector<UniversityDepartment>& graduateInfoList, BTree<int>& graduateInfo) {
    if (!graduateInfoList.empty()) {
        graduateInfoList.clear();
        graduateInfo.clear();
    }

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

    #ifdef DEBUG
        for (int i = 0; i < graduateInfoList.size(); ++i) {
            UniversityDepartment temp = graduateInfoList[i];

            std::cout << i + 1 << ": "
                      << "[" << temp.getOrder() << "]" << " "
                      << temp.getSchoolName() << ", "
                      << temp.getDeptName() << ", "
                      << temp.getDayNight() << ", "
                      << temp.getDegree() << ", "
                      << temp.getStudentNum()
                      << std::endl;
        }

        std::cout << "   ===   \n";
    #endif

    // Build 2-3 tree.
    for (const UniversityDepartment& info : graduateInfoList) {
        graduateInfo.insert(info.getSchoolName(), info.getOrder());
        
        #ifdef DEBUG
            std::cout << "   ===   \n";
            graduateInfo.printBtree();
            std::cout << "   ===   \n";
        #endif
    }

    #ifdef DEBUG
        std::cout << "   ===   \n";
        graduateInfo.printBtree();
        std::cout << "   ===   \n";
        for (const auto& info : graduateInfo.getRootData()) {
            std::cout << info << ", \n";
        }
    #endif

    // Output information.
    std::cout << "Tree height = " << graduateInfo.getHeight() << std::endl;
    printSaveData(graduateInfoList, graduateInfo.getRootData());
}

/** @brief Processes graduate data using a deap. */
void Task2(std::vector<UniversityDepartment>& graduateInfoList, AVLTree<int>& graduateInfo) {
    if (!graduateInfo.isEmpty()) {
        std::cout << "### AVL tree has been built. ###\n";
    } else {
        // Build AVLtree.
        for (const UniversityDepartment& info : graduateInfoList) {
            graduateInfo.insert(info.getDeptName(), info.getOrder());
        }
    }

    // Output information.
    std::cout << "Tree height = " << graduateInfo.getHeight() << std::endl;
    printSaveData(graduateInfoList, graduateInfo.getRootData());
}

int main() {
    int select_command = 0;
    int select_lower_bound = 0;
    int select_upper_bound = 2;
    std::vector<UniversityDepartment> graduateInfoList;
    BTree<int> graduateInfoBTreeOrder3(3, "2-3 tree", "lexicographic");
    AVLTree<int> graduateInfoAVL("lexicographic");

    do {
        while (true) {
            std::cout <<
                "*** Search Tree Utilities **\n"
                "* 0. QUIT                  *\n"
                "* 1. Build 2-3 tree        *\n"
                "* 2. Build AVL tree        *\n"
                "*************************************\n"
                "Input a choice(0, 1, 2): ";

            std::cin >> select_command;

            if (!std::cin.fail()) {
                break;
            } else {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                std::cout << std::endl;
                std::cout << "Command does not exist!\n";
                std::cout << std::endl;  
            }
        }

        switch (select_command) {
        case 0:
            break;
        case 1:
            std::cout << std::endl;
            Task1(graduateInfoList, graduateInfoBTreeOrder3);

            if (!graduateInfoAVL.isEmpty()) { graduateInfoAVL.clear(); }

            std::cout << std::endl;
            break;
        case 2:
            if (graduateInfoList.empty()) {
                std::cout << "### Choose 1 first. ###\n";
            } else {
                Task2(graduateInfoList, graduateInfoAVL);
            }

            std::cout << std::endl;
            break;
        default:
            std::cout << std::endl;
            std::cout << "Command does not exist!\n";
            std::cout << std::endl;
        }
    } while (select_command != 0);

    return 0;
}