/** 
 * @file DS2ex02_27_10927262.cpp
 * @brief A program that utilizes 2-3 trees and AVL trees to efficiently manage and organize graduate student data.
 * @version 1.0.2
 *
 * @details
 * This program implements two different tree data structures, 2-3 trees and AVL trees, for storing and managing graduate student data. 
 * It supports insertion and searching(Root) operations in both tree structures, ensuring efficient data retrieval and organization. 
 * The program is designed to demonstrate the use of self-balancing trees in managing dynamic datasets like student records.
 * The user can perform operations such as adding new graduates, and searching for specific students by various criteria.
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

/**
 * @class BTreeNode
 * @brief Represents a node in a B-Tree.
 * 
 * A node contains keys, associated data, and pointers to child nodes.
 * It also maintains its height.
 * 
 * @tparam T The type of data stored in the node.
 */
template <typename T>
class BTreeNode {
public:
    std::vector<std::string> keys;     // A list of keys in the node, kept sorted.
    std::vector<std::vector<T>> data;  // Data corresponding to each key in the node.
    std::vector<BTreeNode*> children;  // List of pointers to child nodes.
    int height;                        // Height of the node in the tree.

    /**
     * @brief Default constructor initializes a node with height 1.
     */
    BTreeNode() : height(1) {}

    /**
     * @brief Destructor to delete child nodes.
     */
    ~BTreeNode() {
        for (auto child : children) {
            delete child;
        }
    }
};

/**
 * @class BTree
 * @brief Represents a B-Tree.
 * 
 * This class provides methods for inserting keys, managing node splits, 
 * and displaying the structure of the tree.
 * 
 * @tparam T The type of data stored in the tree.
 */
template <typename T>
class BTree {
public:
    /**
     * @brief Constructs a BTree with a specified order and sorting mode.
     * 
     * @param order The order of the tree (defines the maximum number of children per node).
     * @param mode The mode of the tree ("B-tree" or "2-3 tree").
     * @param sort_type The type of sorting ("lexicographic" or "numeric").
     */
    explicit BTree(int order, const std::string& mode = "B-tree", const std::string& sort_type = "lexicographic")
        : order_(order), mode_(mode), sort_type_(sort_type), root_(new BTreeNode<T>) {
        maxKeys_ = getMaxKeys();
    }

    /**
     * @brief Destructor to clean up the tree.
     */
    ~BTree() {
        delete root_;
    }

    /**
     * @brief Clears the tree by deleting the root and creating a new empty tree.
     */
    void clear() {
        delete root_;
        root_ = new BTreeNode<T>();
    }

    /**
     * @brief Inserts a new key-value pair into the B-tree.
     * 
     * @param key The key to insert.
     * @param data The data associated with the key.
     */
    void insert(const std::string& key, const T& data) {
        insertInternal(root_, key, data, maxKeys_);
    
        if (root_->keys.size() > maxKeys_) {  // Check if outof bounds.
            BTreeNode<T>* temp = root_;
            root_ = new BTreeNode<T>();
            root_->children.push_back(temp);
            split(root_, 0, maxKeys_);
        }
    }    
    
    /**
     * @brief Gets the root node of the B-tree.
     * 
     * @return A pointer to the root node.
     */
    BTreeNode<T>* getRoot() const {
        return root_;
    }

    /**
     * @brief Retrieves all data from the root node.
     * 
     * @return A vector containing all the data in the root node.
     */
    std::vector<T> getRootData() const {
        std::vector<T> node_data;
        for (const auto& data_vector : root_->data) {
            if (!data_vector.empty()) {
                for (const auto& data : data_vector) {
                    node_data.push_back(data);
                }
            }
        }

        return node_data;
    }

    /**
     * @brief Gets the height of the B-tree.
     * 
     * @return The height of the root node.
     */
    int getHeight() const {
        return root_->height;  // Directly return the height of the root node.
    }

    /**
     * @brief Prints the structure of the B-tree in a level-by-level format.
     */
    void printBtree() const {
        // Use a queue to implement level-order traversal
        std::vector<std::pair<BTreeNode<T>*, int>> queue;
        queue.emplace_back(root_, 1);  // Start from the root node (height 1)
    
        while (!queue.empty()) {
            BTreeNode<T>* node = queue.front().first;  // The current node being processed
            int level = queue.front().second;          // The height of the current node
            queue.erase(queue.begin());
    
            // Display the level of each node
            std::cout << "Level " << level << ": { ";
    
            // If the node has keys, iterate and display them
            if (!node->keys.empty()) {
                for (size_t i = 0; i < node->keys.size(); ++i) {
                    std::cout << "[" << node->keys[i] << ": (";
                    // If the node has data, iterate and display the data
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
                        std::cout << ", ";  // Separator between keys
                    }
                }
            }
            std::cout << " }" << std::endl;
    
            // Add child nodes to the queue and update the height (level + 1)
            for (BTreeNode<T>* child : node->children) {
                queue.emplace_back(child, level + 1);
            }
        }
    }    

private:
    int order_;              // The order of the tree.
    int maxKeys_;            // Maximum number of keys a node can hold.
    std::string mode_;       // Mode of the tree ("B-tree" or "2-3 tree").
    std::string sort_type_;  // Sorting type ("lexicographic" or "numeric").
    BTreeNode<T>* root_;     // Pointer to the root node.

    /**
     * @brief Determines the maximum number of keys based on the tree mode.
     * 
     * @return The maximum number of keys per node.
     */
    int getMaxKeys() const {
        if (mode_ == "2-3 tree") {
            return 2;  // In a 2-3 tree, each node can hold a maximum of 2 keys
        }
        return 2 * order_ - 1;  // For a standard B-tree
    }

    /**
     * @brief Internal method to insert a key-value pair into the appropriate node.
     * 
     * @param node The current node to insert the key.
     * @param key The key to insert.
     * @param data The data associated with the key.
     * @param maxKeys The maximum number of keys a node can have.
     */
    void insertInternal(BTreeNode<T>* node, const std::string& key, const T& data, int maxKeys_) {
        if (node->children.empty()) {
            // Leaf node
            int index = getChildIndex(node, key);
            if (index < node->keys.size() && node->keys[index] == key) {
                node->data[index].push_back(data);  // If the key exists, just append the data
            } else {
                node->keys.insert(node->keys.begin() + index, key);
                node->data.insert(node->data.begin() + index, std::vector<T>{data});
            }
        } else {
            // Internal node
            int index = getChildIndex(node, key);
            if (index < node->keys.size() && node->keys[index] == key) {
                // If the key exists in an internal node, just append the data
                node->data[index].push_back(data);
            } else {
                // Recursively insert into the child node
                insertInternal(node->children[index], key, data, maxKeys_);
    
                // If the child node splits, check if further handling is needed
                if (node->children[index]->keys.size() > maxKeys_) {
                    split(node, index, maxKeys_);
                    if (compare(node->keys[index], key) < 0) {
                        ++index;  // Update index to ensure insertion goes to the correct child node
                    }
                }
            }
        }
    
        // Update node height
        if (mode_ == "2-3 tree") {
            node->height = 1 + (node->children.empty() ? 0 : node->children[0]->height);
        } else {
            node->height = std::max(node->height, (node->children.empty() ? 1 : node->children[0]->height + 1));
        }
    }    

    /**
     * @brief Splits a node that has exceeded the maximum number of keys.
     * 
     * @param parent The parent node where the split should be registered.
     * @param index The index of the child node being split.
     * @param maxKeys The maximum number of keys per node.
     */
    void split(BTreeNode<T>* parent, int index, int maxKeys_) {
        BTreeNode<T>* nodeToSplit = parent->children[index];
        BTreeNode<T>* newNode = new BTreeNode<T>();
    
        int mid = maxKeys_ / 2;  // For a 2-3 Tree, maxKeys_ is 2, so mid = 1
    
        // Ensure that in 2-3 Tree mode, maxKeys_ must be 2
        if (mode_ == "2-3 tree" && maxKeys_ != 2) {
            delete newNode;
            std::cerr << "Error: In 2-3 tree mode, maxKeys_ must be 2!" << std::endl;
            return;
        }
    
        // Promote the middle key to the parent node
        const std::string& midKey = nodeToSplit->keys[mid];
        auto& midData = nodeToSplit->data[mid];
    
        parent->keys.insert(parent->keys.begin() + index, midKey);
        parent->data.insert(parent->data.begin() + index, midData);
    
        // Move the right half of the keys and data to the new node
        newNode->keys.assign(nodeToSplit->keys.begin() + mid + 1, nodeToSplit->keys.end());
        newNode->data.assign(nodeToSplit->data.begin() + mid + 1, nodeToSplit->data.end());
    
        // Remove the moved keys and data
        nodeToSplit->keys.resize(mid);
        nodeToSplit->data.resize(mid);
    
        // If it's not a leaf node, handle child nodes
        if (!nodeToSplit->children.empty()) {
            newNode->children.assign(nodeToSplit->children.begin() + mid + 1, nodeToSplit->children.end());
            nodeToSplit->children.resize(mid + 1);
        }
    
        // Insert the new node into the parent's child nodes
        parent->children.insert(parent->children.begin() + index + 1, newNode);
    
        // Update the height of the parent node
        parent->height = std::max(parent->children[index]->height, parent->children[index + 1]->height) + 1;
    }     
    


    /**
     * @brief Finds the index of the child node where the key should be inserted.
     * 
     * @param node The current node.
     * @param key The key to insert.
     * @return The index of the child node.
     */
    int getChildIndex(BTreeNode<T>* node, const std::string& key) const {
        int index = 0;
        // Use the comparison function to compare the keys
        while (index < node->keys.size() && compare(node->keys[index], key) < 0) {
            if (compare(node->keys[index], key) == 0) return index;  // Return the index of the existing key
            ++index;
        }
    
        return index;  // Return the index for insertion
    }    

    /**
     * @brief Compares two keys based on the specified sorting type.
     * 
     * @param a The first key.
     * @param b The second key.
     * @return A negative value if a < b, zero if a == b, and a positive value if a > b.
     */
    int compare(const std::string& a, const std::string& b) const {
        if (sort_type_ == "numeric") {
            // Attempt to convert the strings to numbers for comparison
            try {
                int num_a = std::stoi(a);
                int num_b = std::stoi(b);
                return num_a - num_b;
            } catch (const std::invalid_argument&) {
                // If conversion fails, return 0 to avoid errors
                return a.compare(b);
            }
        } else {
            // Default to lexicographical comparison
            return a.compare(b);  // a < b returns -1
        }
    }
};


/**
 * @class AVLTreeNode
 * @brief Represents a node in an AVL Tree.
 * 
 * An AVL tree node contains a key, associated data, pointers to left and right children, and the height of the node.
 * 
 * @tparam T The type of data stored in the node.
 */
template <typename T>
class AVLTreeNode {
public:
    std::string key;      // The key for this node.
    std::vector<T> data;  // The data associated with the key.
    AVLTreeNode* left;    // Pointer to the left child.
    AVLTreeNode* right;   // Pointer to the right child.
    int height;           // The height of the node.

    /**
     * @brief Constructs a new AVLTreeNode with the given key and data.
     * 
     * @param key The key for this node.
     * @param data The data associated with the key.
     */
    AVLTreeNode(const std::string& key, const T& data)
        : key(key), data({data}), left(nullptr), right(nullptr), height(1) {}

    /**
     * @brief Destructor for AVLTreeNode.
     */
    ~AVLTreeNode() {
        left = nullptr;
        right = nullptr;
    }
};

/**
 * @class AVLTree
 * @brief Represents an AVL Tree, a self-balancing binary search tree.
 * 
 * An AVL tree is a binary search tree (BST) where the difference between the heights of the left and right subtrees
 * cannot be more than one for all nodes. If this property is violated, rotations are performed to restore balance.
 * 
 * @tparam T The type of data stored in the tree nodes.
 */
template <typename T>
class AVLTree {
public:
    /**
     * @brief Constructs an AVL Tree with a specified sorting type.
     * 
     * @param sort_type The sorting type, either "lexicographic" or "numeric". Default is "lexicographic".
     */
    explicit AVLTree(const std::string& sort_type = "lexicographic")
        : sort_type_(sort_type), root_(nullptr) {}
    
    /**
     * @brief Destructor to clear the tree and free memory.
     */
    ~AVLTree() {
        clear();
    }

    /**
     * @brief Clears the entire tree by deleting all nodes.
     */    
    void clear() {
        clearInternal(root_);
        root_ = nullptr;
    }

    /**
     * @brief Internal function to recursively clear nodes starting from the given node.
     * 
     * @param node The node to clear.
     */
    void clearInternal(AVLTreeNode<T>* node) {
        if (node == nullptr) return;

        clearInternal(node->left);
        clearInternal(node->right);
        delete node;  // Only delete the node once
    }

    /**
     * @brief Checks if the tree is empty (root is null).
     * 
     * @return True if the tree is empty, false otherwise.
     */
    bool isEmpty() const {
        return root_ == nullptr;
    }

    /**
     * @brief Inserts a new key-value pair into the AVL Tree while maintaining balance.
     * 
     * @param key The key to insert.
     * @param data The data associated with the key.
     */
    void insert(const std::string& key, const T& data) {
        root_ = insertInternal(root_, key, data);
    }

    /**
     * @brief Retrieves the root of the tree.
     * 
     * @return A pointer to the root node.
     */
    AVLTreeNode<T>* getRoot() const {
        return root_;
    }

    /**
     * @brief Retrieves the data of the root node.
     * 
     * @return A vector of data from the root node.
     * @throws std::runtime_error If the tree is empty.
     */
    std::vector<T> getRootData() const {
        if (root_ == nullptr) {
            throw std::runtime_error("Tree root is null");
        }
        return root_->data;
    }

    /**
     * @brief Gets the height of the AVL Tree (height of the root node).
     * 
     * @return The height of the tree.
     */
    int getHeight() const {
        return isEmpty() ? 0 : root_->height;
    }

    /**
     * @brief Prints the structure of the tree in a level-wise manner.
     */
    void printTree() const {
        printTreeInternal(root_, 0);
    }

private:
    std::string sort_type_;  // The sorting type of the tree ("lexicographic" or "numeric").
    AVLTreeNode<T>* root_;   // Pointer to the root node of the AVL Tree.

    /**
     * @brief Internal function that inserts a key-value pair into the tree while maintaining balance.
     * 
     * @param node The current node to insert the key-value pair.
     * @param key The key to insert.
     * @param data The data associated with the key.
     * @return The updated node after insertion.
     */
    AVLTreeNode<T>* insertInternal(AVLTreeNode<T>* node, const std::string& key, const T& data) {
        if (node == nullptr) {
            return new AVLTreeNode<T>(key, data);  // Create a new node
        }

        // Standard BST insertion
        if (compare(key, node->key) < 0) {
            node->left = insertInternal(node->left, key, data);
        } else if (compare(key, node->key) > 0) {
            node->right = insertInternal(node->right, key, data);
        } else {
            // If key is already present, append the new data
            node->data.push_back(data);
            return node;
        }

        // Update height of this ancestor node
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));

        // Get the balance factor and balance the tree
        int balance = getBalance(node);

        // Left Left case
        if (balance > 1 && compare(key, node->left->key) < 0) {
            return rightRotate(node);
        }

        // Right Right case
        if (balance < -1 && compare(key, node->right->key) > 0) {
            return leftRotate(node);
        }

        // Left Right case
        if (balance > 1 && compare(key, node->left->key) > 0) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }

        // Right Left case
        if (balance < -1 && compare(key, node->right->key) < 0) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    /**
     * @brief Performs a left rotation on the given node.
     * 
     * @param x The node to rotate.
     * @return The new root of the rotated subtree.
     */
    AVLTreeNode<T>* leftRotate(AVLTreeNode<T>* x) {
        AVLTreeNode<T>* y = x->right;
        AVLTreeNode<T>* T2 = y->left;

        // Perform rotation
        y->left = x;
        x->right = T2;

        // Update heights
        x->height = 1 + std::max(getHeight(x->left), getHeight(x->right));
        y->height = 1 + std::max(getHeight(y->left), getHeight(y->right));

        return y;  // Return new root
    }

    /**
     * @brief Performs a right rotation on the given node.
     * 
     * @param y The node to rotate.
     * @return The new root of the rotated subtree.
     */
    AVLTreeNode<T>* rightRotate(AVLTreeNode<T>* y) {
        AVLTreeNode<T>* x = y->left;
        AVLTreeNode<T>* T2 = x->right;

        // Perform rotation
        x->right = y;
        y->left = T2;

        // Update heights
        y->height = 1 + std::max(getHeight(y->left), getHeight(y->right));
        x->height = 1 + std::max(getHeight(x->left), getHeight(x->right));

        return x;  // Return new root
    }

    /**
     * @brief Returns the height of the given node.
     * 
     * @param node The node to check the height of.
     * @return The height of the node.
     */
    int getHeight(AVLTreeNode<T>* node) const {
        if (node == nullptr) {
            return 0;
        }
        return node->height;
    }

    /**
     * @brief Returns the balance factor of the given node.
     * 
     * @param node The node to check the balance of.
     * @return The balance factor (height of left subtree - height of right subtree).
     */
    int getBalance(AVLTreeNode<T>* node) const {
        if (node == nullptr) {
            return 0;
        }
        return getHeight(node->left) - getHeight(node->right);
    }

    /**
     * @brief Compares two keys based on the sorting type.
     * 
     * @param a The first key.
     * @param b The second key.
     * @return A negative value if a < b, zero if a == b, and a positive value if a > b.
     */
    int compare(const std::string& a, const std::string& b) const {
        if (sort_type_ == "numeric") {
            // Try converting strings to integers for numeric comparison
            try {
                int num_a = std::stoi(a);
                int num_b = std::stoi(b);
                return num_a - num_b;
            } catch (const std::invalid_argument&) {
                // If conversion fails, fallback to lexicographic comparison
                return a.compare(b);
            }
        } else {
            // Default lexicographic comparison
            return a.compare(b);  // a < b return -1
        }
    }

    /**
     * @brief Internal function to print the tree structure level by level.
     * 
     * @param node The current node to start printing.
     * @param level The current level in the tree.
     */
    void printTreeInternal(AVLTreeNode<T>* node, int level) const {
        if (node == nullptr) {
            return;
        }

        // Print the current level's node
        std::cout << "Level " << level << ": { " << node->key << ": [";
        for (size_t i = 0; i < node->data.size(); ++i) {
            std::cout << node->data[i];
            if (i != node->data.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "] }" << std::endl;

        // Recursively print left and right subtrees
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
    int order_;
    int school_code_;
    std::string school_name_;
    int dept_code_;
    std::string dept_name_;
    std::string day_night_;
    std::string degree_;
    int student_num_;
    int teacher_num_;
    int graduate_num_;
    std::string city_;
    std::string system_type_;

public:
    // Constructor.
    UniversityDepartment(int order, int school_code, const std::string& school_name, int dept_code,
                        const std::string& dept_name, const std::string& day_night,
                        const std::string& degree, int student_num, int teacher_num,
                        int graduate_num, const std::string& city, const std::string& system_type)
        : order_(order),
        school_code_(school_code),
        school_name_(school_name),
        dept_code_(dept_code),
        dept_name_(dept_name),
        day_night_(day_night),
        degree_(degree),
        student_num_(student_num),
        teacher_num_(teacher_num),
        graduate_num_(graduate_num),
        city_(city),
        system_type_(system_type) {}

    // Getters.
    int getOrder() const { return order_; }
    int getSchoolCode() const { return school_code_; }
    const std::string& getSchoolName() const { return school_name_; }
    int getDeptCode() const { return dept_code_; }
    const std::string& getDeptName() const { return dept_name_; }
    const std::string& getDayNight() const { return day_night_; }
    const std::string& getDegree() const { return degree_; }
    int getStudentNum() const { return student_num_; }
    int getTeacherNum() const { return teacher_num_; }
    int getGraduateNum() const { return graduate_num_; }
    const std::string& getCity() const { return city_; }
    const std::string& getSystemType() const { return system_type_; }

    // Setters.
    void setOrder(int order) { order_ = order; }
    void setSchoolCode(int school_code) { school_code_ = school_code; }
    void setSchoolName(const std::string& school_name) { school_name_ = school_name; }
    void setDeptCode(int dept_code) { dept_code_ = dept_code; }
    void setDeptName(const std::string& dept_name) { dept_name_ = dept_name; }
    void setDayNight(const std::string& day_night) { day_night_ = day_night; }
    void setDegree(const std::string& degree) { degree_ = degree; }
    void setStudentNum(int student_num) { student_num_ = student_num; }
    void setTeacherNum(int teacher_num) { teacher_num_ = teacher_num; }
    void setGraduateNum(int graduate_num) { graduate_num_ = graduate_num; }
    void setCity(const std::string& city) { city_ = city; }
    void setSystemType(const std::string& system_type) { system_type_ = system_type; }
};   

/**
 * @brief Removes all non-numeric characters from the given string.
 * 
 * @param input_string The input string to be cleaned.
 * @return A string containing only the numeric characters from the input string.
 */
std::string removeNonNumeric(const std::string& input_string) {
    std::string cleaned_string;
    for (const char& character : input_string) {
        if (std::isdigit(character)) {
            cleaned_string += character;
        }
    }
    return cleaned_string;
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
bool ReadFile(const std::string& input_file_name, std::vector<UniversityDepartment>& graduate_info_list) {
    std::ifstream input_file(input_file_name);

    // Make sure file exists.
    if (!input_file.is_open()) {
        std::cout << std::endl;
        std::cout << "### " << input_file_name << " does not exist! ###" << std::endl;
        std::cout << std::endl;

        // File can't be opened.
        return false;
    }

    // Skip the first three lines (header).
    std::string header_line;
    for (int i = 0; i < 3; ++i) {
        std::getline(input_file, header_line);
    }

    // Parse data from the file.
    std::string line;
    int order = 0;
    while (std::getline(input_file, line)) {
        std::istringstream line_stream(line);
        std::vector<std::string> graduate_information_param;
        std::string token;

        // Separate the data by tabs.
        while (std::getline(line_stream, token, '\t')) {
            graduate_information_param.push_back(token);
        }

        // TODO: When the format does not meet the need (TSV), throw an exception.
        // TODO: If the number of graduate_information_param is not equal to the expected value,
        //       skip the line instead of allowing it to be processed.

        // Make sure data exists.
        if (!graduate_information_param.empty()) {
            ++order;
            int school_code = stoi(removeNonNumeric(graduate_information_param[0]));
            std::string school_name = graduate_information_param[1];
            int dept_code = stoi(removeNonNumeric(graduate_information_param[2]));
            std::string dept_name = graduate_information_param[3];
            std::string day_night = graduate_information_param[4];
            std::string degree = graduate_information_param[5];
            int student_num = stoi(removeNonNumeric(graduate_information_param[6]));
            int teacher_num = stoi(removeNonNumeric(graduate_information_param[7]));
            int graduate_num = stoi(removeNonNumeric(graduate_information_param[8]));
            std::string city = graduate_information_param[9];
            std::string system_type = graduate_information_param[10];

            // Create and store the graduate information.
            UniversityDepartment graduate_info(order, school_code, school_name, dept_code, dept_name,
                                               day_night, degree, student_num, teacher_num,
                                               graduate_num, city, system_type);
            graduate_info_list.push_back(graduate_info);
        }
    }

    input_file.close();

    // File opened and processed successfully.
    return true;
}

/**
 * @brief Prints the saved graduate data in a formatted table.
 * 
 * @param graduateInfoList The list of university department data to print.
 */
void printSaveData(const std::vector<UniversityDepartment>& graduate_info_list, const std::vector<int>& order) {
    std::vector<int> sorted_order = order;  // Create a copy of the order.
    std::sort(sorted_order.begin(), sorted_order.end());  // Sort the copy.

    for (size_t i = 0; i < sorted_order.size(); ++i) {
        int index = sorted_order[i];
        if (index - 1 < 0 || index - 1 >= graduate_info_list.size()) {
            std::cerr << "Index out of range: " << index << std::endl;
            continue;  // Skip invalid indices.
        }

        UniversityDepartment graduate_info = graduate_info_list[index - 1];
        std::cout << i + 1 << ": "
                  << "[" << graduate_info.getOrder() << "] "
                  << graduate_info.getSchoolName() << ", "
                  << graduate_info.getDeptName() << ", "
                  << graduate_info.getDayNight() << ", "
                  << graduate_info.getDegree() << ", "
                  << graduate_info.getStudentNum()
                  << std::endl;
    }

    std::cout << std::endl;
}

/**
 * @brief Task 1: Builds a 2-3 tree from the provided file input.
 * 
 * This function will ask the user for the input file name, read data from it, and build a 2-3 tree (BTree).
 * It will print debug information for each insertion and the final tree structure. 
 * After building the tree, it will output the tree height and other relevant data.
 * 
 * @param graduate_info_list A vector that stores information about the graduate departments.
 * @param graduate_info The 2-3 tree (BTree) to store the department data.
 */
void Task1(std::vector<UniversityDepartment>& graduate_info_list, BTree<int>& graduate_info) {
    // If there is any existing data, clear both the list and the tree
    if (!graduate_info_list.empty()) {
        graduate_info_list.clear();
        graduate_info.clear();
    }

    // Continue to ask the user to enter the file number until they choose to exit.
    while (true) {
        std::string input_file_name = "";

        std::cout << "Input a file number ([0] Quit): ";
        std::cin >> input_file_name;

        if ("0" == input_file_name) {
            return;  // Exit if the user enters 0.
        } else {
            input_file_name = "input" + input_file_name + ".txt";

            // Try to open the file.
            // If it fails, prompt for the file name again.
            if (!ReadFile(input_file_name, graduate_info_list)) continue;

            // Check if the file contains any data.
            if (!graduate_info_list.empty()) {
                break;  // Successfully read data, exit the loop.
            } else {
                std::cout << std::endl;
                std::cout << "### Get nothing from the file " << input_file_name << " ! ###" << std::endl;
                return;
            }
        }
    }

    // Debug mode: Print the contents of the graduate_info_list
    #ifdef DEBUG
        for (int i = 0; i < graduate_info_list.size(); ++i) {
            UniversityDepartment temp = graduate_info_list[i];

            std::cout << i + 1 << ": "
                      << "[" << temp.getOrder() << "]"
                      << " " << temp.getSchoolName() << ", "
                      << temp.getDeptName() << ", "
                      << temp.getDayNight() << ", "
                      << temp.getDegree() << ", "
                      << temp.getStudentNum()
                      << std::endl;
        }

        std::cout << "   ===   \n";
    #endif

    // Build 2-3 tree.
    for (const UniversityDepartment& info : graduate_info_list) {
        graduate_info.insert(info.getSchoolName(), info.getOrder());

        // Debug mode: Print the current state of the tree after each insertion
        #ifdef DEBUG
            std::cout << "   ===   \n";
            graduate_info.printBtree();
            std::cout << "   ===   \n";
        #endif
    }

    // Debug mode: Print the final state of the tree and the root data
    #ifdef DEBUG
        std::cout << "   ===   \n";
        graduate_info.printBtree();
        std::cout << "   ===   \n";
        for (const auto& info : graduate_info.getRootData()) {
            std::cout << info << ", \n";
        }
    #endif

    // Output information.
    std::cout << "Tree height = " << graduate_info.getHeight() << std::endl;
    printSaveData(graduate_info_list, graduate_info.getRootData());
}

/**
 * @brief Task 2: Builds an AVL tree from the provided department information.
 * 
 * This function builds an AVL tree if it hasn't been created already. It checks if the tree is empty and inserts data.
 * After the tree is built, it outputs the tree height and relevant data.
 * 
 * @param graduate_info_list A vector containing the department information.
 * @param graduate_info The AVL tree to store the department data.
 */
void Task2(std::vector<UniversityDepartment>& graduate_info_list, AVLTree<int>& graduate_info) {
    if (!graduate_info.isEmpty()) {
        std::cout << "### AVL tree has been built. ###\n";
    } else {
        // Build AVL tree.
        for (const UniversityDepartment& info : graduate_info_list) {
            graduate_info.insert(info.getDeptName(), info.getOrder());
        }
    }

    // Output information.
    std::cout << "Tree height = " << graduate_info.getHeight() << std::endl;
    printSaveData(graduate_info_list, graduate_info.getRootData());
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
    std::vector<UniversityDepartment> graduate_info_list;
    BTree<int> graduate_info_b_tree_order_3(3, "2-3 tree", "lexicographic");
    AVLTree<int> graduate_info_avl("lexicographic");

    do {
        while (true) {
            // Display the menu options for the user
            std::cout <<
                "*** Search Tree Utilities **\n"
                "* 0. QUIT                  *\n"
                "* 1. Build 2-3 tree        *\n"
                "* 2. Build AVL tree        *\n"
                "*************************************\n"
                "Input a choice(0, 1, 2): ";

            std::cin >> select_command;

            // Check if the input is valid
            if (!std::cin.fail()) {
                break;
            } else {
                // Handle invalid input
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                std::cout << std::endl;
                std::cout << "Command does not exist!\n";
                std::cout << std::endl;
            }
        }

        // Handle the different options based on the user's choice
        switch (select_command) {
        case 0:
            break;
        case 1:
            std::cout << std::endl;
            // Call Task 1 to build the 2-3 tree
            Task1(graduate_info_list, graduate_info_b_tree_order_3);

            // If AVL tree has data, clear it before building the new tree
            if (!graduate_info_avl.isEmpty()) { graduate_info_avl.clear(); }

            std::cout << std::endl;
            break;
        case 2:
            if (graduate_info_list.empty()) {
                // If no data has been read, prompt the user to choose Task 1 first
                std::cout << "### Choose 1 first. ###\n";
            } else {
                // Call Task 2 to build the AVL tree
                Task2(graduate_info_list, graduate_info_avl);
            }

            std::cout << std::endl;
            break;
        default:
            std::cout << std::endl;
            std::cout << "Command does not exist!\n";
            std::cout << std::endl;
        }
    } while (select_command != 0); // Continue until the user selects option 0 (quit)

    return 0;
}