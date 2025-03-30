```plantuml
@startuml
class HeapBase {
    # data: std::vector<std::pair<int, T>>
    {abstract} # heapifyUp(index: int): void
    {abstract} # heapifyDown(index: int): void
    + isEmpty(): bool
    + size(): int
    + clear(): void
    + top(): const std::pair<int, T>&
    + bottom(): const std::pair<int, T>&
    + leftmostBottom(): const std::pair<int, T>&
    + push(key: int, value: T): void
    + pop(): std::pair<int, T>
    + height(): int
    + height(nodeNum: int): int
    + replaceAt(index: int, newKey: int, newValue: T): std::pair<int, T>
    + getAt(index: int): const std::pair<int, T>&
    + isFullBinaryTree(nodeCount: int): bool
    + printHeap(): void
}

class MinHeap<T> {
    # heapifyUp(index: int): void {override}
    # heapifyDown(index: int): void {override}
}

class MaxHeap<T> {
    # heapifyUp(index: int): void {override}
    # heapifyDown(index: int): void {override}
}

class MinMaxHeap<T> {
    # isMinLevel(index: int): bool
    # heapifyUp(index: int): void
    # heapifyUpMin(index: int): void
    # heapifyUpMax(index: int): void
    # heapifyDown(index: int): void
    # heapifyDownMin(index: int): void
    # heapifyDownMax(index: int): void
    + getMin(): std::pair<int, T>
    + getMax(): std::pair<int, T>
    + popMin(): std::pair<int, T>
    + popMax(): std::pair<int, T>
}

class Deap<T> {
    - minHeap: MinHeap<T>
    - maxHeap: MaxHeap<T>
    + isEmpty(): bool
    + clear(): void
    + topMin(): const std::pair<int, T>&
    + topMax(): const std::pair<int, T>&
    + bottom(): const std::pair<int, T>&
    + leftmostBottom(): const std::pair<int, T>&
    + push(key: int, data: T): void
    + popMin(): std::pair<int, T>
    + popMax(): std::pair<int, T>
    + printHeaps(): void
}

HeapBase <|-- MinHeap
HeapBase <|-- MaxHeap
HeapBase <|-- MinMaxHeap
Deap *-- MinHeap
Deap *-- MaxHeap
@enduml