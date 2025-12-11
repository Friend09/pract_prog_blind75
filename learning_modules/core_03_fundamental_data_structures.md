# CORE_03: Fundamental Data Structures

## Overview

### What Are Data Structures?

**Data structures** are ways to organize and store data so it can be accessed and modified efficiently. The choice of data structure often determines the efficiency of your algorithm.

```
┌─────────────────────────────────────────────────────────────┐
│            DATA STRUCTURES IN ONE SENTENCE                  │
├─────────────────────────────────────────────────────────────┤
│  "The RIGHT data structure can transform an O(n²) solution  │
│   into an O(n) solution."                                   │
└─────────────────────────────────────────────────────────────┘
```

### Why Data Structures Matter

| Without Right Structure | With Right Structure |
|------------------------|---------------------|
| Linear search: O(n) | Binary search tree: O(log n) |
| Find duplicates: O(n²) | Hash set: O(n) |
| Get min element: O(n) | Heap: O(1) |
| FIFO processing: complex | Queue: simple |

---

## Arrays

### What Is an Array?

An **array** is a contiguous block of memory storing elements of the same type, accessed by index.

```
Memory layout:
┌─────┬─────┬─────┬─────┬─────┐
│  10 │  20 │  30 │  40 │  50 │
└─────┴─────┴─────┴─────┴─────┘
  [0]   [1]   [2]   [3]   [4]

Address calculation:
address(arr[i]) = base_address + (i × element_size)
```

### Array Characteristics

| Property | Description |
|----------|-------------|
| **Contiguous** | Elements stored next to each other in memory |
| **Indexed** | O(1) access to any element by position |
| **Fixed size** | Static arrays can't change size |
| **Same type** | All elements have same data type |

### Array Operations and Complexity

```python
# Python lists are dynamic arrays
arr = [10, 20, 30, 40, 50]

# ACCESS - O(1)
element = arr[2]  # → 30

# SEARCH (unsorted) - O(n)
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# INSERT at end - O(1) amortized
arr.append(60)  # [10, 20, 30, 40, 50, 60]

# INSERT at position - O(n) [must shift elements]
arr.insert(2, 25)  # [10, 20, 25, 30, 40, 50, 60]
#                        ↑ shift everything right

# DELETE at end - O(1)
arr.pop()  # Remove last element

# DELETE at position - O(n) [must shift elements]
arr.pop(2)  # Remove element at index 2
#              ↑ shift everything left
```

### Array Complexity Summary

| Operation | Time | Notes |
|-----------|------|-------|
| Access by index | O(1) | Direct memory calculation |
| Search (unsorted) | O(n) | Must check each element |
| Search (sorted) | O(log n) | Binary search |
| Insert at end | O(1)* | Amortized for dynamic arrays |
| Insert at middle | O(n) | Must shift elements |
| Delete at end | O(1) | No shifting needed |
| Delete at middle | O(n) | Must shift elements |

### Common Array Patterns

```python
# Two-pointer technique
def reverse_in_place(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

# Sliding window
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Prefix sum
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix
    # Range sum [i, j] = prefix[j+1] - prefix[i]
```

---

## Linked Lists

### What Is a Linked List?

A **linked list** is a sequence of nodes where each node contains data and a pointer to the next node.

```
Singly Linked List:
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│ val: 10   │    │ val: 20   │    │ val: 30   │    │ val: 40   │
│ next: ─────────│ next: ─────────│ next: ─────────│ next: None │
└───────────┘    └───────────┘    └───────────┘    └───────────┘
     ↑
   head

Doubly Linked List:
       ←─────────────────────────────────────────→
┌───────────┐    ┌───────────┐    ┌───────────┐
│ prev:None │    │ prev: ────│    │ prev: ────│
│ val: 10   │    │ val: 20   │    │ val: 30   │
│ next: ────│────│ next: ────│────│ next:None │
└───────────┘    └───────────┘    └───────────┘
     ↑                                  ↑
   head                               tail
```

### Linked List Implementation

```python
class ListNode:
    """Node for singly linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class DoublyListNode:
    """Node for doubly linked list."""
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

### Linked List Operations

```python
# TRAVERSAL - O(n)
def traverse(head):
    current = head
    while current:
        print(current.val)
        current = current.next

# SEARCH - O(n)
def search(head, target):
    current = head
    while current:
        if current.val == target:
            return current
        current = current.next
    return None

# INSERT at head - O(1)
def insert_at_head(head, val):
    new_node = ListNode(val)
    new_node.next = head
    return new_node  # New head

# INSERT at tail - O(n) [or O(1) with tail pointer]
def insert_at_tail(head, val):
    new_node = ListNode(val)
    if not head:
        return new_node
    
    current = head
    while current.next:
        current = current.next
    current.next = new_node
    return head

# DELETE node - O(1) if have reference, O(n) to find
def delete_node(head, target):
    dummy = ListNode(0, head)
    prev = dummy
    current = head
    
    while current:
        if current.val == target:
            prev.next = current.next
            break
        prev = current
        current = current.next
    
    return dummy.next

# REVERSE - O(n) time, O(1) space
def reverse(head):
    prev = None
    current = head
    
    while current:
        next_temp = current.next  # Save next
        current.next = prev       # Reverse pointer
        prev = current            # Move prev forward
        current = next_temp       # Move current forward
    
    return prev  # New head
```

### Linked List vs Array Comparison

| Operation | Array | Linked List |
|-----------|-------|-------------|
| Access by index | O(1) | O(n) |
| Search | O(n) | O(n) |
| Insert at beginning | O(n) | O(1) |
| Insert at end | O(1)* | O(n)** |
| Insert at middle | O(n) | O(1)*** |
| Delete | O(n) | O(1)*** |
| Memory | Contiguous | Scattered |
| Extra memory | None | Pointer per node |

*Amortized  **O(1) with tail pointer  ***After finding position

### Common Linked List Patterns

```python
# Dummy head technique (simplifies edge cases)
def remove_elements(head, val):
    dummy = ListNode(0, head)
    prev = dummy
    current = head
    
    while current:
        if current.val == val:
            prev.next = current.next
        else:
            prev = current
        current = current.next
    
    return dummy.next

# Fast and slow pointers (find middle)
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # Middle node

# Cycle detection (Floyd's algorithm)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

---

## Stacks

### What Is a Stack?

A **stack** is a LIFO (Last In, First Out) data structure. Think of a stack of plates—you can only add or remove from the top.

```
Stack operations:

    PUSH            POP            PEEK
      ↓              ↑              👁
    ┌───┐         ┌───┐         ┌───┐
    │ 4 │ ←top    │   │         │ 4 │ ←top (return 4)
    ├───┤         ├───┤         ├───┤
    │ 3 │         │ 3 │ ←top    │ 3 │
    ├───┤         ├───┤         ├───┤
    │ 2 │         │ 2 │         │ 2 │
    ├───┤         ├───┤         ├───┤
    │ 1 │         │ 1 │         │ 1 │
    └───┘         └───┘         └───┘
```

### Stack Implementation

```python
# Using Python list (recommended)
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):      # O(1) amortized
        self.items.append(item)
    
    def pop(self):             # O(1)
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):            # O(1)
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):        # O(1)
        return len(self.items) == 0
    
    def size(self):            # O(1)
        return len(self.items)

# Usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())   # 3
print(stack.peek())  # 2
```

### Stack Operations Complexity

| Operation | Time | Description |
|-----------|------|-------------|
| Push | O(1)* | Add to top |
| Pop | O(1) | Remove from top |
| Peek | O(1) | View top element |
| isEmpty | O(1) | Check if empty |
| Size | O(1) | Get number of elements |

*Amortized for dynamic arrays

### Classic Stack Problems

```python
# Valid Parentheses
def is_valid(s: str) -> bool:
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    
    return len(stack) == 0

# Min Stack (get minimum in O(1))
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val
    
    def get_min(self):
        return self.min_stack[-1]

# Evaluate Reverse Polish Notation
def eval_rpn(tokens: list[str]) -> int:
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b)  # Truncate toward zero
    }
    
    for token in tokens:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))
    
    return stack[0]
```

### When to Use Stacks

- **Matching**: Parentheses, brackets, tags
- **Reversal**: Reverse strings, reverse linked lists
- **History**: Undo functionality, browser back button
- **Parsing**: Expression evaluation, syntax parsing
- **DFS**: Iterative depth-first search
- **Monotonic**: Next greater element, histogram problems

---

## Queues

### What Is a Queue?

A **queue** is a FIFO (First In, First Out) data structure. Think of a line at a store—first person in line is first to be served.

```
Queue operations:

   ENQUEUE (rear)              DEQUEUE (front)
        ↓                           ↓
    ┌───┬───┬───┬───┬───┐     ┌───┬───┬───┬───┐
    │ 1 │ 2 │ 3 │ 4 │ 5 │  →  │ 2 │ 3 │ 4 │ 5 │  (1 removed)
    └───┴───┴───┴───┴───┘     └───┴───┴───┴───┘
      ↑               ↑         ↑           ↑
   front           rear      front        rear
```

### Queue Implementation

```python
from collections import deque

# Using deque (recommended - O(1) operations on both ends)
class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):    # O(1)
        self.items.append(item)
    
    def dequeue(self):          # O(1)
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")
    
    def front(self):            # O(1)
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")
    
    def is_empty(self):         # O(1)
        return len(self.items) == 0
    
    def size(self):             # O(1)
        return len(self.items)

# Why not use list?
# list.pop(0) is O(n) because it shifts all elements
# deque.popleft() is O(1)
```

### Queue Operations Complexity

| Operation | Time | Description |
|-----------|------|-------------|
| Enqueue | O(1) | Add to rear |
| Dequeue | O(1)* | Remove from front |
| Front/Peek | O(1) | View front element |
| isEmpty | O(1) | Check if empty |
| Size | O(1) | Get number of elements |

*O(1) with deque, O(n) with list

### Queue Variants

```python
# Double-ended Queue (Deque)
from collections import deque

dq = deque()
dq.append(1)      # Add to right
dq.appendleft(0)  # Add to left
dq.pop()          # Remove from right
dq.popleft()      # Remove from left

# Priority Queue (Heap-based)
import heapq

pq = []
heapq.heappush(pq, (1, "low priority"))
heapq.heappush(pq, (3, "high priority"))
heapq.heappush(pq, (2, "medium"))
print(heapq.heappop(pq))  # (1, "low priority")

# Circular Queue (fixed size)
class CircularQueue:
    def __init__(self, k: int):
        self.queue = [None] * k
        self.size = k
        self.front = 0
        self.rear = -1
        self.count = 0
    
    def enqueue(self, value: int) -> bool:
        if self.is_full():
            return False
        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = value
        self.count += 1
        return True
    
    def dequeue(self) -> bool:
        if self.is_empty():
            return False
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return True
    
    def is_empty(self) -> bool:
        return self.count == 0
    
    def is_full(self) -> bool:
        return self.count == self.size
```

### Classic Queue Problems

```python
# BFS using queue
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

# Level order traversal of binary tree
def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### When to Use Queues

- **BFS**: Breadth-first search traversal
- **Level-order**: Tree level-by-level processing
- **Scheduling**: Task scheduling, print queues
- **Buffering**: Data streams, message queues
- **Sliding window**: With deque for O(1) operations on both ends

---

## Comparison: Stack vs Queue

```
Stack (LIFO)                    Queue (FIFO)
═══════════════                 ═══════════════

    ┌───┐                       ┌───┬───┬───┐
    │ 3 │ ← push/pop            │ 1 │ 2 │ 3 │
    ├───┤                       └───┴───┴───┘
    │ 2 │                         ↑       ↑
    ├───┤                      dequeue   enqueue
    │ 1 │
    └───┘

Use cases:                      Use cases:
• Undo/Redo                     • BFS
• Parentheses matching          • Level-order traversal
• DFS (iterative)               • Task scheduling
• Expression evaluation         • Buffer/Stream processing
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              FUNDAMENTAL DATA STRUCTURES                    │
├─────────────────────────────────────────────────────────────┤
│ ARRAY:                                                      │
│ • O(1) access by index                                      │
│ • O(n) insert/delete at middle                              │
│ • Contiguous memory                                         │
├─────────────────────────────────────────────────────────────┤
│ LINKED LIST:                                                │
│ • O(1) insert/delete at known position                      │
│ • O(n) access by index                                      │
│ • Non-contiguous memory                                     │
├─────────────────────────────────────────────────────────────┤
│ STACK (LIFO):                                               │
│ • O(1) push, pop, peek                                      │
│ • Use for: DFS, matching, reversal, history                 │
├─────────────────────────────────────────────────────────────┤
│ QUEUE (FIFO):                                               │
│ • O(1) enqueue, dequeue, front                              │
│ • Use for: BFS, level-order, scheduling                     │
├─────────────────────────────────────────────────────────────┤
│ TIPS:                                                       │
│ • Use deque for O(1) queue operations                       │
│ • Use dummy nodes to simplify linked list edge cases        │
│ • Stack → DFS; Queue → BFS                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Arrays
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Two Sum | Hash map lookup | 1 |
| 2 | Best Time to Buy/Sell Stock | Track minimum | 121 |
| 3 | Contains Duplicate | Set for O(1) lookup | 217 |
| 4 | Product of Array Except Self | Prefix/suffix products | 238 |
| 5 | Maximum Subarray | Kadane's algorithm | 53 |

### Linked Lists
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Reverse Linked List | Three-pointer technique | 206 |
| 2 | Merge Two Sorted Lists | Dummy head | 21 |
| 3 | Linked List Cycle | Fast/slow pointers | 141 |
| 4 | Remove Nth Node From End | Two pointers with gap | 19 |
| 5 | Middle of Linked List | Fast/slow pointers | 876 |

### Stacks
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Valid Parentheses | Matching pairs | 20 |
| 2 | Min Stack | Auxiliary min stack | 155 |
| 3 | Daily Temperatures | Monotonic stack | 739 |
| 4 | Evaluate Reverse Polish | Operator stack | 150 |

### Queues
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Number of Islands | BFS traversal | 200 |
| 2 | Binary Tree Level Order | Level processing | 102 |
| 3 | Rotting Oranges | Multi-source BFS | 994 |
| 4 | Design Circular Queue | Circular array | 622 |

---

## Summary

1. **Arrays**: Fast access, slow insert/delete. Use when you need random access.

2. **Linked Lists**: Fast insert/delete, slow access. Use when order matters and frequent modifications.

3. **Stacks**: LIFO for DFS, matching, reversal. Use when you need to process most recent first.

4. **Queues**: FIFO for BFS, level-order. Use when you need to process in arrival order.

5. **Choose wisely**: The right data structure can transform your algorithm's efficiency.
