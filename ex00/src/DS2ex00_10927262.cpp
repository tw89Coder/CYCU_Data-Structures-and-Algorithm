/** 
 * @file DS2ex00_10927262.cpp
 * @brief A program that uses Binary Search Tree (BST) to manage and filter and delete graduate data.
 * @version 1.0.0
 *
 * @details
 * This program provides functionalities to read and process graduate data stored in files
 * using a Binary Search Tree (BST) as the underlying data structure. The main features include:
 * - Read data from the file and construct BST.
 * - Filter according to the threshold and output the filtered graduate information.
 * - Delete a specific graduate from the BST.
 *
 * The program supports input files formatted in Tab-separated values (TSV) and organizes
 * the data in the BST with graduate numbers as the key.
 *
 * @author 
 * - 10927262 呂易鴻
 */

 #include <algorithm>
 #include <fstream>
 #include <functional>
 #include <iomanip>
 #include <iostream>
 #include <limits>
 #include <mutex>
 #include <queue>
 #include <sstream>
 #include <stdexcept>
 #include <string>
 #include <thread>
 #include <vector>

 /** 
 * @class Node
 *
 * @brief A generic node structure for the binary tree.
 *
 * @tparam T The type of the elements stored in the node (e.g., int, float).
 */
template <typename T>
struct Node {
    int key;
    std::vector<T> data;
    Node* left;
    Node* right;
    int height;

    Node(int k, const T& value) : key(k), left(nullptr), right(nullptr), height(1) {
        data.push_back(value);
    }
};

/**
 * @class BinaryTree
 *
 * @brief A generic binary tree implementation.
 *
 * This class implements a binary tree based on the Abstract Data Type (ADT) defined in
 * Chapter 15 of "Data Abstraction & Problem Solving with C++" (7th edition) by Frank M. Carrano and Timothy Henry.
 * Specifically, it follows the Binary Tree interface outlined on page 439 of the book.
 * In addition to the standard methods from the textbook's Binary Tree interface, 
 * this class includes custom methods to enhance functionality, such as printTree(), getMaxKeyVal(), findGreaterOrEqual(), findGreaterOrEqualParallel(), etc.
 * 
 * @see Carrano, F. M., & Henry, T. (2015). Data Abstraction & Problem Solving with C++ (7th ed.), Chapter 15.
 *
 * @tparam T The type of the elements stored in the tree (e.g., int, float).
 */ 
template <typename T>
class BinaryTree {
private:
    Node<T>* root; // Root node of the binary tree.
    std::mutex result_mutex;

    /**
     * @brief Get the height of a node.
     *
     * @param node The node to get the height of.
     * @return The height of the node.
     */
    int getHeight(Node<T>* node) const {
        return node ? node->height : 0;
    }

    /**
     * @brief Get the number of nodes in the tree.
     *
     * @param node The root node of the subtree.
     * @return The number of nodes in the subtree.
     */
    int getNodeCount(Node<T>* node) const {
        if (!node) return 0;
        return 1 + getNodeCount(node->left) + getNodeCount(node->right);
    }

    /**
     * @brief Recursively deletes all nodes in the tree.
     *
     * @param node Reference to the pointer of the current subtree root.
     */
    void clearTree(Node<T>*& node) {
        if (node) {
            clearTree(node->left);
            clearTree(node->right);
            delete node;
            node = nullptr;
        }
    }

    /**
     * @brief Inserts a new node into the binary search tree.
     *
     * @details
     * -If a node with the given key already exists, the data is appended to
     *  the existing node's data vector. This operation maintains the tree's
     *  binary search structure.
     *
     * @param node Pointer to the root of the current subtree.
     * @param key The key of the node to insert.
     * @param newData The data to associate with the key.
     * @param success Boolean flag indicating whether the insertion succeeded.
     * @return Node<T>* Updated root of the subtree after insertion.
     */
    Node<T>* addNode(Node<T>* node, int key, const T& newData, bool& success) {
        // If the node is NULL, a new node is created and returned.
        if (!node) {
            success = true;
            return new Node<T>(key, newData);
        }

        if (key < node->key) {
            // If the key is less than the current node's key, insert the new data in the left subtree.
            node->left = addNode(node->left, key, newData, success);
        } else if (key > node->key) {
            // If the key is greater than the current node's key, insert the new data in the right subtree.
            node->right = addNode(node->right, key, newData, success);
        } else {
            // If key is equal, insert newData into the data vector.
            node->data.push_back(newData);
            success = true; // No error for duplicate keys, just added data.
        }

        // Update the height of the current node.
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));

        // Return the updated node.
        return node;
    }

    /**
     * @brief Removes a node with the specified key from the binary search tree.
     *
     * This function recursively searches for the node with the given key in the BST.
     * If the key is found, the node is removed, and the BST structure is maintained.
     * - If the node has one child, the child replaces the node.
     * - If the node has two children, the in-order successor (smallest node in the
     *   right subtree) replaces the node.
     *
     * @param node Pointer to the current subtree root.
     * @param key The key of the node to remove.
     * @param success A reference to a boolean indicating whether the removal was successful.
     * @return Node<T>* Updated root of the subtree after removal.
     *
     * @details
     * - Updates the height of each ancestor node after removal.
     * - If the node with the specified key is not found, `success` is set to `false`.
     * - Frees the memory of the removed node to prevent memory leaks.
     */
    Node<T>* removeNode(Node<T>* node, int key, bool& success) {
        // If the node is NULL, the key wasn't found. Set success to false.
        if (!node) {
            success = false;
            return nullptr;
        }
        
        // The key is less than the current node's key, search in the left subtree.
        if (key < node->key) {
            node->left = removeNode(node->left, key, success);
        } else if (key > node->key) {
            // The key is greater than the current node's key, search in the right subtree.
            node->right = removeNode(node->right, key, success);
        } else {
            // The node with the key is found.
            success = true;

            // Case 1: The node has one or no children, so remove the node.
            if (!node->left || !node->right) {
                // If there's a child, set temp to that child.
                Node<T>* temp = node->left ? node->left : node->right;
                // Delete the current node
                delete node;
                // Return the child node to replace the current node.
                return temp;
            } else {
                // Case 2: The node has two children, so find the successor (the smallest node in the right subtree).
                Node<T>* successor = getMinNode(node->right);
                // Replace the current node's key and data with the successor's key and data.
                node->key = successor->key;
                node->data = successor->data;
                // Recursively remove the successor from the right subtree.
                node->right = removeNode(node->right, successor->key, success);
            }
        }

        // Update the height of the current node.
        if (node) {
            // Based on the height of its children.
            node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));
        }

        // Return the updated node.
        return node;
    }

    /**
     * @brief Finds the node with the minimum key in the subtree.
     *
     * @param node Pointer to the root of the subtree.
     * @return Node<T>* Pointer to the node with the minimum key, or nullptr if the subtree is empty.
     */
    Node<T>* getMinNode(Node<T>* node) const {
        while (node && node->left) {
            node = node->left;
        }

        return node;
    }

    /**
     * @brief Finds the node with the maximum key in the subtree.
     *
     * @param node Pointer to the root of the subtree.
     * @return Node<T>* Pointer to the node with the minimum key, or nullptr if the subtree is empty.
     */
    Node<T>* getMaxNode(Node<T>* node) const {
        while (node && node->right) {
            node = node->right;
        }

        return node;
    }

    /**
     * @brief Finds the node with the specified key in the subtree.
     *
     * @param node Pointer to the root of the subtree.
     * @param key The key of the node to find.
     * @return Node<T>* Pointer to the node with the minimum key, or nullptr if the subtree is empty.
     */
    Node<T>* findNode(Node<T>* node, int key) const {
        if (!node || node->key == key) return node;
        if (key < node->key) return findNode(node->left, key);
        return findNode(node->right, key);
    }

    /**
     * @brief Traverses the tree in pre-order and applies the visit function to each node.
     *
     * @param node Pointer to the root of the subtree.
     * @param visit Function to apply to each node.
     */
    void traversePreOrder(Node<T>* node, const std::function<void(const Node<T>*)>& visit) const {
        if (node) {
            visit(node);
            traversePreOrder(node->left, visit);
            traversePreOrder(node->right, visit);
        }
    }

    /**
     * @brief Traverses the tree in in-order and applies the visit function to each node.
     *
     * @param node Pointer to the root of the subtree.
     * @param visit Function to apply to each node.
     */
    void traverseInOrder(Node<T>* node, const std::function<void(const Node<T>*)>& visit) const {
        if (node) {
            traverseInOrder(node->left, visit);
            visit(node);
            traverseInOrder(node->right, visit);
        }
    }

    /**
     * @brief Traverses the tree in post-order and applies the visit function to each node.
     *
     * @param node Pointer to the root of the subtree.
     * @param visit Function to apply to each node.
     */
    void traversePostOrder(Node<T>* node, const std::function<void(const Node<T>*)>& visit) const {
        if (node) {
            traversePostOrder(node->left, visit);
            traversePostOrder(node->right, visit);
            visit(node);
        }
    }

    /**
    * @brief Recursively prints the structure of the binary search tree.
    *
    * This function provides a detailed view of the BST structure, including the
    * level of each node, its key, and its height. It prints both to the console
    * and to a file when used with an `std::ofstream` object.
    *
    * @param node Pointer to the current subtree root.
    * @param output Output stream (e.g., `std::cout` or `std::ofstream`) to write the tree structure.
    * @param level The depth level of the current node in the tree (default is 0).
    * @param prefix A string used to format the output, indicating the parent-child relationship.
    *
    * @details
    * - Nodes are labeled with `L->` for left children and `R->` for right children.
    * - If a child node is null, outputs "NULL" for that branch.
    * - Can handle arbitrarily deep trees by recursively traversing left and right subtrees.
    */
    void printTreeHelper(Node<T>* node, std::ostream& output, int level = 0, const std::string& prefix = "") const {
        if (node != nullptr) {
            // Output the current node's level, key, and height
            output << prefix << "Level " << level << ": Key = " << node->key << ", Height = " << node->height << std::endl;

            // If the node has left or right children, continue to recursively print them
            if (node->left || node->right) {
                // If the left child exists, recursively print the left subtree; otherwise, print "NULL"
                if (node->left) {
                    printTreeHelper(node->left, output, level + 1, prefix + "L->");
                } else {
                    output << prefix + "L-> NULL" << std::endl;
                }

                // If the right child exists, recursively print the right subtree; otherwise, print "NULL"
                if (node->right) {
                    printTreeHelper(node->right, output, level + 1, prefix + "R->");
                } else {
                    output << prefix + "R-> NULL" << std::endl;
                }
            }
        }
    }

    /**
     * @brief Prints the tree structure in a level-order format with parentheses.
     * 
     * @details
     * This function performs a level-order traversal (Breadth-First Search) of the binary tree.
     * For each level, it outputs all nodes in the following format: 
     * 
     * - (key, index1,index2,...)
     * - Nodes at the same level are separated by a single space.
     * - Each level is prefixed with "<Level N>" where N represents the current level.
     * - If the node has no children, it will not enqueue further nodes.
     *
     * @param node The root node of the binary tree to be printed.
     * 
     * @tparam T The type of data stored in each node. It must support the `getIndex()` method.
     */
    void printTreeParenthesesHelper(Node<T>* node) const {
        if (node == nullptr) return;

        std::cout << "HP tree:" << std::endl;

        std::queue<std::pair<Node<T>*, int>> nodeQueue;  // Store level and node.
        int currentLevel = 1;  // Initial level.
        nodeQueue.push({node, currentLevel});

        while (!nodeQueue.empty()) {
            int levelNodeCount = nodeQueue.size();  // record the amount of nodes in current level 
            std::cout << "<level " << currentLevel << "> ";

            for (int i = 0; i < levelNodeCount; ++i) {
                Node<T>* currentNode = nodeQueue.front().first;
                int nodeLevel = nodeQueue.front().second;
                nodeQueue.pop();

                // Output node's level and data.
                std::cout << "(" << currentNode->key;
                if (!currentNode->data.empty()) {
                    std::cout << ", ";
                    for (size_t j = 0; j < currentNode->data.size(); j++) {
                        std::cout << currentNode->data[j].getIndex();
                        if (j != currentNode->data.size() - 1) {  // If data is the last element.
                            std::cout << ",";
                        }
                    }
                }
                std::cout << ")";

                // Check if it's the last element of this level.
                if (i != levelNodeCount - 1) {
                    std::cout << " "; 
                }

                // Add node to the queue and mark the level.
                if (currentNode->left) {
                    nodeQueue.push({currentNode->left, nodeLevel + 1});
                }
                if (currentNode->right) {
                    nodeQueue.push({currentNode->right, nodeLevel + 1});
                }
            }

            currentLevel++;  // Go to next level.
            std::cout << std::endl;
        }
    }

    /**
    * @brief Finds all nodes with keys greater than or equal to a specified threshold.
    *
    * This function recursively traverses the tree, collecting data from all nodes
    * where the key is greater than or equal to the threshold. It also counts the
    * total number of visited nodes during the traversal.
    *
    * @param node Pointer to the current subtree root.
    * @param threshold The minimum key value to include in the result.
    * @param result A vector to store the data from matching nodes.
    * @return int The total number of nodes visited during the traversal.
    *
    * @details
    * - Adds the data from matching nodes (keys >= threshold) to the `result` vector.
    * - Traverses the left subtree only if the current node's key is strictly greater
    *   than the threshold (to find smaller potential matches).
    * - Always traverses the right subtree to ensure all larger keys are explored.
    * - Returns the total count of nodes visited, which can be used for performance analysis.
    */
    int findGreaterOrEqualHelper(Node<T>* node, int threshold, std::vector<T>& result) {
        // TODO: Consider introducing multiple threads to spread the search and sorting workload.
        if (!node) return 0; // If the node is null, return 0 as the number of visited nodes.

        int visited_count = 1; // Count the current node as visited.

        if (node->key >= threshold) {
            // If the current node's key is greater than or equal to the threshold,
            // add the node's data to the result vector.
            result.insert(result.end(), node->data.begin(), node->data.end());
        }

        if (node->key > threshold) {
            // If the current node's key is greater than the threshold,
            // continue searching the left subtree.
            visited_count += findGreaterOrEqualHelper(node->left, threshold, result);
        }

        // Always continue searching the right subtree, regardless of the current node's key.
        visited_count += findGreaterOrEqualHelper(node->right, threshold, result);

        // Return the total number of visited nodes.
        return visited_count; 
    }

    /**
    * @brief Finds all nodes with keys less than or equal to a specified threshold.
    *
    * This function recursively traverses the tree, collecting data from all nodes
    * where the key is less than or equal to the given threshold. It also counts the
    * total number of visited nodes during the traversal.
    *
    * @param node Pointer to the current subtree root.
    * @param threshold The maximum key value to include in the result.
    * @param result A vector to store the data from matching nodes.
    * @return int The total number of nodes visited during the traversal.
    *
    * @details
    * - Adds the data from matching nodes (keys ≤ threshold) to the `result` vector.
    * - Always traverses the left subtree to find smaller potential matches.
    * - Traverses the right subtree only if the current node's key is ≤ threshold,
    *   ensuring all valid keys are explored.
    * - Returns the total count of nodes visited, which can be useful for performance analysis.
    */
    int findLessOrEqualHelper(Node<T>* node, int threshold, std::vector<T>& result) {
        // TODO: Consider introducing multiple threads to spread the search and sorting workload.
        if (!node) return 0; // If the node is null, return 0 as the number of visited nodes.

        int visited_count = 1; // Count the current node as visited.

        if (node->key <= threshold) {
            // If the current node's key is less than or equal to the threshold,
            // add the node's data to the result vector.
            result.insert(result.end(), node->data.begin(), node->data.end());
        }

        if (node->key < threshold) {
             // If the current node's key is within the threshold, continue searching the right subtree.
            visited_count += findLessOrEqualHelper(node->right, threshold, result);
        }

        // Always search the left subtree for smaller or equal keys.
        visited_count += findLessOrEqualHelper(node->left, threshold, result);

        // Return the total number of visited nodes.
        return visited_count; 
    }

    /**
    * @brief A helper function for parallel subtree traversal.
    *        This function is designed to be executed by multiple threads, allowing for parallelized searching.
    * 
    * @param node The root node of the current subtree being processed.
    * @param threshold The threshold value used to filter nodes based on their key.
    * @param shared_result A shared vector to store the results of nodes that meet the threshold condition.
    * @param visited_count A reference to the counter tracking the number of visited nodes.
    *                      This is incremented for every node that is traversed.
    */
    void parallelSearch(Node<T>* node, int threshold, std::vector<T>& shared_result, int& visited_count) {
        // Return immediately if the node is null (base case).
        if (!node) return;

        // Increment the visited_count in a thread-safe manner using a lock.
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            visited_count++;
        }

        // If the current node's key meets or exceeds the threshold, add its data to the shared_result vector.
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            if (node->key >= threshold) {
                shared_result.insert(shared_result.end(), node->data.begin(), node->data.end());
            }
        }

        // Recursively process the left subtree if the current node's key is greater than the threshold.
        if (node->key > threshold) {
            parallelSearch(node->left, threshold, shared_result, visited_count);
        }

        // Always process the right subtree.
        parallelSearch(node->right, threshold, shared_result, visited_count);
    }

public:
    /**
     * @brief Constructs an empty binary search tree (BST).
     */
    BinaryTree() : root(nullptr) {}

    /**
     * @brief Destructor that clears the tree and releases all allocated memory.
     *
     * This function ensures no memory leaks occur by calling the `clear()` method.
     */
    ~BinaryTree() {
        clear();
    }

    /**
     * @brief Checks if the tree is empty.
     *
     * @return `true` if the tree is empty, false otherwise.
     */
    bool isEmpty() const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        return root == nullptr;
    }

    /**
     * @brief Gets the height of the binary search tree.
     *
     * @return The height of the tree. Returns 0 if the tree is empty.
     */
    int getHeight() const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        return getHeight(root);
    }

    /**
     * @brief Gets the total number of nodes in the tree.
     *
     * @return The number of nodes in the tree. Returns 0 if the tree is empty.
     */
    int getNumberOfNodes() const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        return getNodeCount(root);
    }

    /**
     * @brief Gets the data stored in the root node of the tree.
     *
     * @return A constant reference to the data vector of the root node.
     * @throws std::runtime_error if the tree is empty.
     */
    const std::vector<T>& getRootData() const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        if (!root) {
            throw std::runtime_error("Tree is empty.");
        }

        return root->data;
    }

    /**
     * @brief Updates the data in the root node of the tree.
     *
     * If the tree is empty, a new root node is created with a default key of 0.
     *
     * @param newData The new data to set in the root node.
     */
    void setRootData(const T& newData) {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        if (!root) {
            root = new Node<T>(0, newData); // Default key is 0.
        } else {
            root->data.clear();
            root->data.push_back(newData);
        }
    }

    /**
     * @brief Adds a new node with the specified key and data to the tree.
     *
     * If a node with the given key already exists, the new data is added to its data vector.
     *
     * @param key The key of the node to add.
     * @param newData The data to associate with the key.
     * @return `true` if the node was successfully added or updated, false otherwise.
     */
    bool add(int key, const T& newData) {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        bool success = false;
        root = addNode(root, key, newData, success);

        return success;
    }

    /**
     * @brief Removes the node with the specified key from the tree.
     *
     * If the node has two children, its in-order successor replaces it.
     *
     * @param key The key of the node to remove.
     * @return `true` if the node was successfully removed, false otherwise.
     */
    bool remove(int key) {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        bool success = false;
        root = removeNode(root, key, success);

        return success;
    }

    /**
     * @brief Clears all nodes in the tree, releasing allocated memory.
     *
     * After calling this method, the tree becomes empty.
     */
    void clear() {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        clearTree(root);
    }

    /**
     * @brief Gets the data associated with the specified key in the tree.
     *
     * @param key The key of the node to retrieve.
     * @return A constant reference to the data vector of the node.
     * @throws std::runtime_error if the key is not found.
     */
    const std::vector<T>& getEntry(int key) const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        Node<T>* node = findNode(root, key);
        if (!node) {
            throw std::runtime_error("Entry not found.");
        }

        return node->data;
    }

    /**
     * @brief Finds the maximum key in the tree.
     *
     * @return The maximum key value.
     * @throws std::runtime_error if the tree is empty.
     */
    int getMaxKeyVal() const {
        Node<T>* node = getMaxNode(root);
        if (!node) {
            throw std::runtime_error("Max key not found.");
        }

        return node->key;
    }

    /**
     * @brief Checks if a node with the specified key exists in the tree.
     *
     * @param key The key to search for.
     * @return `true` if the key exists in the tree, false otherwise.
     */
    bool contains(int key) const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        return findNode(root, key) != nullptr;
    }

    /**
     * @brief Performs a pre-order traversal of the tree.
     *
     * Visits the root node first, followed by the left and right subtrees.
     *
     * @param visit A callback function to process each visited node.
     *
     * Example usage:
     * @code
     * // Pre-order traversal with std::cout to print node keys
     * BinaryTree<graduate> pokemon_guide;
     * std::function<void(const Node<graduate>*)> visit = [](const Node<graduate>* node) {
     *     std::cout << node->key << " ";
     * };
     *
     * pokemon_guide.preorderTraverse(visit);
     * @endcode
     */
    void preorderTraverse(const std::function<void(const Node<T>*)>& visit) const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        traversePreOrder(root, visit);
    }

    /**
     * @brief Performs an in-order traversal of the tree.
     *
     * Visits the left subtree first, followed by the root node, then the right subtree.
     *
     * @param visit A callback function to process each visited node.
     *
     * Example usage:
     * @code
     * // In-order traversal with std::cout to print node keys
     * BinaryTree<graduate> pokemon_guide;
     * std::function<void(const Node<graduate>*)> visit = [](const Node<graduate>* node) {
     *     std::cout << node->key << " ";
     * };
     *
     * pokemon_guide.inorderTraverse(visit);
     * @endcode
     */
    void inorderTraverse(const std::function<void(const Node<T>*)>& visit) const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        traverseInOrder(root, visit);
    }

    /**
     * @brief Performs a post-order traversal of the tree.
     *
     * Visits the left and right subtrees first, followed by the root node.
     *
     * @param visit A callback function to process each visited node.
     *
     * Example usage:
     * @code
     * // Post-order traversal with std::cout to print node keys
     * BinaryTree<graduate> pokemon_guide;
     * std::function<void(const Node<graduate>*)> visit = [](const Node<graduate>* node) {
     *     std::cout << node->key << " ";
     * };
     *
     * pokemon_guide.postorderTraverse(visit);
     * @endcode
     */
    void postorderTraverse(const std::function<void(const Node<T>*)>& visit) const {
        // Implementation based on Carrano & Henry's Binary Tree ADT
        traversePostOrder(root, visit);
    }

    /**
     * @brief Prints the tree structure to both the console and a file.
     *
     * The output includes node levels, keys, and heights. The file output is saved
     * as "tree_output.txt".
     *
     * Example output:
     * @code
     * Level 0: Key = 45, Height = 5
     * L->Level 1: Key = 44, Height = 2
     * L->L->Level 2: Key = 40, Height = 1
     * L->R-> NULL
     * R->Level 1: Key = 60, Height = 4
     * R->L->Level 2: Key = 59, Height = 2
     * R->L->L->Level 3: Key = 50, Height = 1
     * R->L->R-> NULL
     * R->R->Level 2: Key = 80, Height = 3
     * R->R->L->Level 3: Key = 78, Height = 2
     * R->R->L->L->Level 4: Key = 65, Height = 1
     * R->R->L->R->Level 4: Key = 79, Height = 1
     * R->R->R-> NULL
     * @endcode
     */
    void printTree() const {
        // Terminal output
        printTreeHelper(root, std::cout, 0);

        // File output
        std::ofstream outfile("tree_output.txt");
        if (outfile.is_open()) {
            printTreeHelper(root, outfile, 0);
            outfile.close();
        }
    }

    /**
     * @brief Prints the tree structure in a level-order format with parentheses.
     * 
     * @details
     * This function calls the helper function `printTreeParenthesesHelper` to 
     * perform a level-order traversal (Breadth-First Search) of the binary tree.
     * The tree is printed level by level in the following format:
     * 
     * - Each level starts with "<Level N>", where N is the level number.
     * - Each node is displayed in parentheses as "(key, index1,index2,...)".
     * - Multiple nodes in the same level are printed on the same line, 
     *   with a space separating each node.
     * 
     * Example output:
     * @code
     * HP tree:
     * <Level 1>(56, 8)
     * <Level 2>(44, 7)(78, 6)
     * <Level 3>(40, 13)(45, 1,10,14)(60, 2)(79, 9)
     * <Level 4>(50, 11)(65, 15)(80, 3)
     * @endcode
     */
    void printTreeParentheses() const {
        printTreeParenthesesHelper(root);
    }

    /**
     * @brief Finds all nodes with keys greater than or equal to the specified threshold.
     *
     * @param threshold The minimum key value to include in the result.
     * @param result A vector to store the data from matching nodes.
     * @return The total number of nodes visited during the traversal.
     */
    int findGreaterOrEqual(int threshold, std::vector<T>& result) {
        return findGreaterOrEqualHelper(root, threshold, result);
    }

    /**
     * @brief Finds all nodes with keys less than or equal to the specified threshold.
     *
     * @param threshold The maximum key value to include in the result.
     * @param result A vector to store the data from matching nodes.
     * @return The total number of nodes visited during the traversal.
     */
     int findLessOrEqual(int threshold, std::vector<T>& result) {
        return findLessOrEqualHelper(root, threshold, result);
    }

    /**
     * @brief Finds all nodes with keys greater than or equal to the specified threshold.
     *
     * This function uses multithreading to accelerate the search process by concurrently
     * traversing the left and right subtrees. It efficiently skips irrelevant nodes
     * in the tree to improve performance.
     *
     * @param threshold The minimum key value to include in the result.
     *                  Nodes with keys greater than or equal to this value will be included.
     * @param result A reference to a vector that will store the data from matching nodes.
     *               The function appends the matching node data into this vector.
     * @return The total number of nodes visited during the traversal.
     *         This count includes nodes from the left and right subtrees, as well as nodes
     *         directly traversed in the main thread.
     *
     * @details
     * - The traversal begins from the root and skips over nodes whose keys are less than the threshold
     *   by following the right child path until a valid starting node is found.
     * - If the current node's key is exactly equal to the threshold, it is immediately processed.
     * - If the current node's key exceeds the threshold:
     *     - A new thread is created to process the left subtree (if it exists and is relevant).
     *     - Another thread is created to process the right subtree (if it exists).
     * - The main thread processes the current node and waits for both threads to complete.
     * - Thread-safe operations are ensured by synchronizing access to shared counters and result vectors.
     * - Once all threads finish execution, the results from the left and right subtrees are merged
     *   into the final result vector.
     */
    int findGreaterOrEqualParallel(int threshold, std::vector<T>& result) {
        // Return 0 if the tree is empty.
        if (!root) return 0;

        std::vector<std::thread> threads;           // Vector to store threads for parallel execution.
        std::vector<T> left_result, right_result;   // Separate vectors to store results from left and right subtrees.
        int visited_count = 0, left_count = 0, right_count = 0; // Counters to track the number of visited nodes.

        Node<T>* current_node = root;

        // Traverse down the right side of the tree until a node meets the threshold condition.
        // The main target is to find key values that are greater than the threshold.
        while (current_node && current_node->key <= threshold) {
            visited_count++;
            if (current_node->key == threshold) {
                result.insert(result.end(), current_node->data.begin(), current_node->data.end());
            }

            // If no right child exists, stop traversal.
            if (!current_node->right) break;
            // Move to the right child node.
            current_node = current_node->right;
        }

        // Create threads to process the left and right subtrees in parallel.
        // Each thread calls the parallelSearch method to traverse the subtree,
        // passing the reference to the corresponding result vector and counter.
        if (current_node->left && current_node->key > threshold) {
            // If the left child exists and the current node's key exceeds the threshold
            threads.emplace_back(
                &BinaryTree::parallelSearch, // Function pointer to the parallelSearch method.
                this,                            // Pass the current BinaryTree instance(parallelSearch).
                current_node->left,              // Start the search from the left child.
                threshold, 
                std::ref(left_result),           // Reference to the left result vector.
                std::ref(left_count)
            );
        }

        if (current_node->right) { 
            // If the right child exists, create a new thread to traverse the right subtree.
            threads.emplace_back(
                &BinaryTree::parallelSearch, 
                this,
                current_node->right, 
                threshold, 
                std::ref(right_result), 
                std::ref(right_count)
            );
        }

        // Process the root node in the main thread.
        // If the root node's key meets the threshold, add its data to the result vector,
        // and increment the visited_count.
        if (current_node->key > threshold) {
            result.insert(result.end(), current_node->data.begin(), current_node->data.end());
            visited_count++;
        }

        // Wait for all threads to finish their execution.
        for (auto& t : threads) {
            t.join();
        }

        // Merge the results from the left and right subtrees into the main result vector.
        result.insert(result.end(), left_result.begin(), left_result.end());
        result.insert(result.end(), right_result.begin(), right_result.end());

        // Update the total visited node count by adding the counts from the left and right subtrees.
        visited_count += left_count + right_count;

        // Return the total number of visited nodes.
        return visited_count;
    }
};

/**
 * @class UniversityDepartment
 *
 * @brief Represents a graduate with various attributes like hp, types, and generation.
 *
 * This class provides getter and setter methods for all its attributes:
 * - Getter: Retrieve the value of an attribute.
 * - Setter: Modify the value of an attribute.
 * 
 * Example usage:
 * @code
 * graduate pikachu;
 * pikachu.setName("Pikachu");
 * pikachu.setHp(35);
 * std::cout << pikachu.getName() << " has " << pikachu.getHp() << " HP.";
 * @endcode
 */
class UniversityDepartment {
    private:
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
        UniversityDepartment(int schoolCode, const std::string& schoolName, int deptCode, const std::string& deptName,
                             const std::string& dayNight, const std::string& degree, int studentNum, int teacherNum, 
                             int graduateNum, const std::string& city, const std::string& systemType)
            : schoolCode(schoolCode), schoolName(schoolName), deptCode(deptCode), deptName(deptName), 
              dayNight(dayNight), degree(degree), studentNum(studentNum), teacherNum(teacherNum), 
              graduateNum(graduateNum), city(city), systemType(systemType) {}
    
        // Getters
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
 * @brief Reads graduate data from a tab-separated file.
 *
 * This function reads data from the specified input file and populates the given vector with graduate objects.
 * The input file is expected to be in TSV (Tab-Separated Values) format, where each row contains
 * the details of a graduate.
 *
 * @param inputFileName The name of the input file to read.
 * @param graduateInformationList A reference to a vector where graduate objects will be stored.
 * @return `true` if the file is successfully read, `false` if the file cannot be opened.
 *
 * @details
 * - The first line of the file is assumed to be a header and is skipped.
 * - If a row is invalid, it will be ignored.
 */
 bool ReadFile(const std::string& inputFileName, std::vector<UniversityDepartment>& graduateInformationList){
    std::ifstream input_file(inputFileName);

    // Make sure file exist.
    if (!input_file.is_open()) {
        std::cout << std::endl;
        std::cout << "### " << inputFileName << " does not exist! ###" << std::endl;
        std::cout << std::endl;

        // File can't be opened.
        return false;
    }

    // Skip head of three lines.
    std::string header_line;
    std::getline(input_file, header_line);
    std::getline(input_file, header_line);
    std::getline(input_file, header_line);
    
    // Get data.
    std::string line;
    while (std::getline(input_file, line)) {
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
            int schoolCode = stoi(graduateInformationParam[0]);
            std::string schoolName = graduateInformationParam[1];
            int deptCode = stoi(graduateInformationParam[2]);
            std::string deptName = graduateInformationParam[3];
            std::string dayNight = graduateInformationParam[4];
            std::string degree = graduateInformationParam[5];
            int studentNum = stoi(graduateInformationParam[6]);
            int teacherNum = stoi(graduateInformationParam[7]);
            int graduateNum = stoi(graduateInformationParam[8]);
            std::string city = graduateInformationParam[9];
            std::string systemType = graduateInformationParam[10];

            // Create and store the graduateInformation.
            UniversityDepartment graduateInformation(schoolCode, schoolName, deptCode, deptName, dayNight, degree, studentNum, teacherNum, graduateNum, city, systemType);
            graduateInformationList.push_back(graduateInformation);
        }
    }

    input_file.close();

    // Opened file success.
    return true;
}

/**
 * @brief Sorts a vector of graduate objects by their school code values in descending order.
 *
 * If two graduate have the same school code, they are sorted by their index in ascending order.
 *
 * @param graduateInformationList A reference to a vector of graduate objects to be sorted.
 *
 * @details
 * - This function uses `std::sort` with a custom comparison lambda function.
 * - Sorting is performed in-place, modifying the original vector.
 * - Lower school code values appear first.
 */
 void sortGraduateInfoList(std::vector<UniversityDepartment>& graduateInformationList) {
    std::sort(graduateInformationList.begin(), graduateInformationList.end(),
        [](const UniversityDepartment& a, const UniversityDepartment& b) {
            return a.getSchoolCode() < b.getSchoolCode();
        }
    );
}

/**
 * @brief Prints the saved graduate data in a formatted table.
 * 
 * @param graduateInformationList The list of university department data to print.
 */
void printSaveData(std::vector<UniversityDepartment>& graduateInformationList) {
    for (int i = 0; i < graduateInformationList.size(); ++i) {
        UniversityDepartment graduateInfo = graduateInformationList[i];

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

/**
 * @brief Reads graduate data from a file, sorts it, prints it, and builds a binary search tree.
 * 
 * @param graduateInformation The binary search tree to store the graduate data.
 */
void Task1(BinaryTree<UniversityDepartment>& graduateInformation) {
    std::vector<UniversityDepartment> graduateInformationList;

    // Clear the BST if it's not empty before inserting new data.
    if (!graduateInformation.isEmpty()) graduateInformation.clear();

    // Continue to ask the user to enter the file number until the user chooses to exit.
    while (true) {
        std::string inputFileName = "";

        std::cout << "Input a file number: ";
        std::cin >> inputFileName;

        if (inputFileName == "0") {
            return; // Exit if the user enters 0.
        } else {
            inputFileName = "input" + inputFileName + ".txt";

            // Try to open the file.
            // If fail, enter the file name again.
            if (!ReadFile(inputFileName, graduateInformationList)) continue;

            // Check if the file contains any data.
            if(!graduateInformationList.empty()) {
                break; // Read data success, jump out the loop.
            } else {
                std::cout << std::endl;
                std::cout << "### Get nothing from the file "<< inputFileName <<" ! ###" << std::endl;
                return;
            }
        }
    }

    // Sort the data in descending order by school code.
    sortGraduateInfoList(graduateInformationList);
    // Print out the list of input files (7 items).
    printSaveData(graduateInformationList);

    // Build binary tree.
    for (const UniversityDepartment& graduateInfo : graduateInformationList) {
        graduateInformation.add(graduateInfo.getGraduateNum(), graduateInfo);
    }

    // Print out the tree height.
    std::cout << "Tree height {Number of gradutes} = " << graduateInformation.getHeight() << std::endl;


    // Debug output of the tree structure.
    // Can be used with tree_pic.py to generate a binary tree diagram.
    // Use -DDEBUG to compile.
    #ifdef DEBUG
    pokemon_guide.printTree();
    #endif
}
 
/**
 * @brief Removes nodes with a graduate number less than or equal to a user-defined threshold.
 * 
 * @param graduateInformation The binary search tree containing graduate data.
 */
void Task2(BinaryTree<UniversityDepartment>& graduateInformation) {
    long long int threshold = 0; // When I use 999999999999999999, example demo can work.
    bool isFirstTry = true;
    int maxKeyVal = graduateInformation.getMaxKeyVal();

    // Continue to require the user to enter the threshold until the input is legal.
    while(true) {
        std::cout << (isFirstTry ? "Input the number of graduates: " : "Try again: ");
        std::cin >> threshold;
        isFirstTry = false;

        // Check if the input is a positive integer within the valid range.
        if (std::cin.fail()) {
            // Clear the error status.
            std::cin.clear(); 
            // Ignore incorrect input.
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
            std::cout << std::endl;
            std::cout << "### It is NOT a positive integer. ###" << std::endl;
        } else if (threshold < 0) {
            std::cout << std::endl;
            std::cout << "### It is NOT a positive integer. ###" << std::endl;
        } else if (threshold < 1 || threshold > maxKeyVal) {
            std::cout << std::endl;
            std::cout << "### It is NOT in [1," << maxKeyVal << "]. ###" << std::endl;
        } else {
            break; // Legal input, jump out the loop.
        }
    }

    std::vector<UniversityDepartment> graduateInformationList; // Save all matching items.
    
    // Do search.
    int node_count = graduateInformation.findLessOrEqual(threshold, graduateInformationList);
    // Print out the list of input files (7 items).
    std::cout << "Deleted records:" << std::endl;
    printSaveData(graduateInformationList);
    // Delete the nodes.
    for (const UniversityDepartment& graduateInfo : graduateInformationList) {
        graduateInformation.remove(graduateInfo.getGraduateNum());
    }

    // Print out the tree height.
    std::cout << "Tree height {Number of gradutes} = " << graduateInformation.getHeight() << std::endl;
}

/**
 * @brief Main function to run the University Graduate Information System.
 * 
 * @return int Exit status of the program.
 */
int main() {
    int select_num = 0;
    int select_lower_bound = 0;
    int select_upper_bound = 2;
    BinaryTree<UniversityDepartment> graduateInformation;

    do {
        while (true) {
            std::cout <<
                "*** University Graduate Information System ***\n"
                "* 0. Quit                                    *\n"
                "* 1. Create Binary Search Trees              *\n"
                "* 2. Removal by Number of Graduates          *\n"
                "**********************************************\n"
                "Input a command(0, 1, 2): ";

            std::cin >> select_num;

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

        switch (select_num) {
        case 0:
            break;
        case 1:
            std::cout << std::endl;
            Task1(graduateInformation);
            std::cout << std::endl;
            break;
        case 2:
            if (graduateInformation.isEmpty()) {
                std::cout << std::endl;
                std::cout << "----- Execute Mission 1 first! -----" << std::endl;
            } else {
                Task2(graduateInformation);
            }
            
            std::cout << std::endl;
            break;
        default:
            std::cout << std::endl;
            std::cout << "Command does not exist!" << std::endl;
            std::cout << std::endl;
        }
    } while (select_num != 0);

    return 0;
}