# CORE_05: Searching Algorithms

## Overview

### Why Searching Matters

Searching is one of the most common operations in programming. The right search algorithm can make the difference between O(n) and O(log n)—which at scale means milliseconds vs. hours.

```
┌─────────────────────────────────────────────────────────────┐
│              SEARCHING IMPACT AT SCALE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  n = 1 billion elements                                     │
│                                                             │
│  Linear Search: ~1 billion operations                       │
│  Binary Search: ~30 operations    (log₂(10⁹) ≈ 30)          │
│                                                             │
│  That's 33 MILLION times faster!                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Search Algorithm Categories

```
SEARCHING ALGORITHMS
│
├── LINEAR STRUCTURES
│   ├── Linear Search: O(n) - unsorted data
│   ├── Binary Search: O(log n) - sorted arrays
│   └── Interpolation Search: O(log log n) avg - uniform distribution
│
├── TREE STRUCTURES
│   ├── BST Search: O(log n) avg, O(n) worst
│   ├── Balanced BST: O(log n) guaranteed
│   └── Trie Search: O(m) where m = key length
│
├── HASH-BASED
│   └── Hash Table Lookup: O(1) average
│
└── GRAPH SEARCH
    ├── BFS: O(V + E) - shortest path in unweighted
    └── DFS: O(V + E) - path existence, components
```

---

## Linear Search

### Basic Linear Search

**Idea**: Check each element one by one until found or end reached.

```python
def linear_search(arr: list, target) -> int:
    """
    Search for target in unsorted array.
    
    Time:  O(n) - must check each element
    Space: O(1)
    
    Use when: Data is unsorted, small dataset, one-time search
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# With early termination for sorted array
def linear_search_sorted(arr: list, target) -> int:
    """
    Search in sorted array - can stop early.
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
        if arr[i] > target:  # Gone past where target would be
            break
    return -1
```

### Linear Search Trace

```
Array: [4, 2, 7, 1, 9, 3, 6]
Target: 9

Step 1: Check arr[0]=4, 4≠9, continue
Step 2: Check arr[1]=2, 2≠9, continue
Step 3: Check arr[2]=7, 7≠9, continue
Step 4: Check arr[3]=1, 1≠9, continue
Step 5: Check arr[4]=9, 9=9, FOUND! Return 4
```

### When to Use Linear Search

| Situation | Why Linear Search |
|-----------|-------------------|
| Unsorted data | Binary search requires sorted |
| Small dataset (n < 20) | Overhead of binary search not worth it |
| One-time search | Not worth sorting first |
| Linked list | No random access for binary search |

---

## Binary Search

### The Core Algorithm

**Idea**: Repeatedly divide the search space in half.

**Prerequisite**: Array MUST be sorted!

```python
def binary_search(arr: list, target) -> int:
    """
    Search for target in SORTED array.
    
    Time:  O(log n) - halve search space each iteration
    Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1   # Target in right half
        else:
            right = mid - 1  # Target in left half
    
    return -1  # Not found
```

### Binary Search Trace

```
Array: [1, 3, 5, 7, 9, 11, 13, 15]
Target: 9

Step 1: left=0, right=7, mid=3
        arr[3]=7 < 9, search right half
        left = 4

Step 2: left=4, right=7, mid=5
        arr[5]=11 > 9, search left half
        right = 4

Step 3: left=4, right=4, mid=4
        arr[4]=9 == 9, FOUND! Return 4

Visual:
[1, 3, 5, 7, 9, 11, 13, 15]
 L        M            R    → 7 < 9, go right

[1, 3, 5, 7, 9, 11, 13, 15]
             L   M      R   → 11 > 9, go left

[1, 3, 5, 7, 9, 11, 13, 15]
             LMR            → 9 = 9, found!
```

### Why Binary Search Is O(log n)

```
n elements → n/2 → n/4 → n/8 → ... → 1

How many halvings until we reach 1?

n / 2^k = 1
n = 2^k
k = log₂(n)

Examples:
n = 16:   16→8→4→2→1  = 4 steps  = log₂(16)
n = 1024: 10 steps = log₂(1024)
n = 1M:   20 steps = log₂(1,000,000)
```

### Recursive Binary Search

```python
def binary_search_recursive(arr: list, target, left: int = 0, right: int = None) -> int:
    """
    Recursive implementation.
    
    Time:  O(log n)
    Space: O(log n) - recursion stack
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

---

## Binary Search Variations

### Finding the Leftmost (First) Occurrence

```python
def find_first(arr: list, target) -> int:
    """
    Find first occurrence of target in sorted array with duplicates.
    
    Key insight: When we find target, keep searching LEFT
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid      # Record this position
            right = mid - 1   # But keep looking left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Example: [1, 2, 2, 2, 3, 4]
# find_first(arr, 2) → 1 (first occurrence)
```

### Finding the Rightmost (Last) Occurrence

```python
def find_last(arr: list, target) -> int:
    """
    Find last occurrence of target.
    
    Key insight: When we find target, keep searching RIGHT
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid      # Record this position
            left = mid + 1    # But keep looking right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### Finding Insertion Point (Lower Bound)

```python
def lower_bound(arr: list, target) -> int:
    """
    Find first position where target could be inserted.
    Returns index of first element >= target.
    
    Equivalent to: bisect.bisect_left(arr, target)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

# Example: [1, 3, 5, 5, 7]
# lower_bound(arr, 5) → 2 (first 5)
# lower_bound(arr, 4) → 2 (where 4 would go)
# lower_bound(arr, 8) → 5 (after all elements)
```

### Finding Upper Bound

```python
def upper_bound(arr: list, target) -> int:
    """
    Find first position AFTER where target could be inserted.
    Returns index of first element > target.
    
    Equivalent to: bisect.bisect_right(arr, target)
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left

# Example: [1, 3, 5, 5, 7]
# upper_bound(arr, 5) → 4 (after last 5)
```

### Search in Rotated Sorted Array

```python
def search_rotated(nums: list, target: int) -> int:
    """
    Search in array that was sorted, then rotated.
    Example: [4, 5, 6, 7, 0, 1, 2]
    
    Key insight: One half is always sorted.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

### Binary Search on Answer (Search Space)

```python
def min_capacity(weights: list, days: int) -> int:
    """
    Find minimum ship capacity to ship all packages in 'days' days.
    
    Binary search on the ANSWER, not the input!
    Search space: [max(weights), sum(weights)]
    """
    def can_ship(capacity):
        """Check if we can ship with given capacity."""
        day_count = 1
        current = 0
        
        for weight in weights:
            if current + weight > capacity:
                day_count += 1
                current = 0
            current += weight
        
        return day_count <= days
    
    left = max(weights)      # Minimum possible: largest single package
    right = sum(weights)     # Maximum possible: all in one day
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid      # Try smaller capacity
        else:
            left = mid + 1   # Need larger capacity
    
    return left
```

---

## Python's bisect Module

```python
import bisect

arr = [1, 3, 5, 5, 7, 9]

# Find insertion point for maintaining sorted order
bisect.bisect_left(arr, 5)   # 2 (first 5)
bisect.bisect_right(arr, 5)  # 4 (after last 5)
bisect.bisect(arr, 5)        # Same as bisect_right: 4

# Insert while maintaining sorted order
bisect.insort_left(arr, 4)   # arr = [1, 3, 4, 5, 5, 7, 9]
bisect.insort_right(arr, 6)  # arr = [1, 3, 4, 5, 5, 6, 7, 9]

# Find element (manual)
def binary_search_bisect(arr, target):
    idx = bisect.bisect_left(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1
```

---

## Search in Trees

### Binary Search Tree (BST) Search

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def bst_search(root: TreeNode, target: int) -> TreeNode:
    """
    Search in Binary Search Tree.
    
    Time:  O(h) where h = tree height
           O(log n) for balanced, O(n) for skewed
    Space: O(1) iterative
    
    BST Property: left < node < right
    """
    current = root
    
    while current:
        if current.val == target:
            return current
        elif target < current.val:
            current = current.left
        else:
            current = current.right
    
    return None

# Recursive version
def bst_search_recursive(root: TreeNode, target: int) -> TreeNode:
    if not root or root.val == target:
        return root
    
    if target < root.val:
        return bst_search_recursive(root.left, target)
    else:
        return bst_search_recursive(root.right, target)
```

### BST Search Trace

```
BST:
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13

Search for 7:
1. Start at 8: 7 < 8, go left
2. At 3: 7 > 3, go right
3. At 6: 7 > 6, go right
4. At 7: FOUND!

Steps: 4 (height of tree)
```

### Balanced vs Unbalanced BST

```
BALANCED (height = log n)        UNBALANCED (height = n)

        4                         1
       / \                         \
      2   6                         2
     / \ / \                         \
    1  3 5  7                         3
                                       \
                                        4
                                         \
                                          5

Search time: O(log n)            Search time: O(n)
```

---

## Search in Graphs

### Breadth-First Search (BFS)

```python
from collections import deque

def bfs(graph: dict, start, target) -> bool:
    """
    Search level by level. Finds shortest path in unweighted graph.
    
    Time:  O(V + E)
    Space: O(V)
    """
    visited = {start}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        if node == target:
            return True
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False

# Find shortest path (number of edges)
def bfs_shortest_path(graph: dict, start, target) -> int:
    visited = {start}
    queue = deque([(start, 0)])  # (node, distance)
    
    while queue:
        node, dist = queue.popleft()
        
        if node == target:
            return dist
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # Not reachable
```

### Depth-First Search (DFS)

```python
def dfs(graph: dict, start, target, visited=None) -> bool:
    """
    Search by going deep first.
    
    Time:  O(V + E)
    Space: O(V) for recursion stack
    """
    if visited is None:
        visited = set()
    
    if start == target:
        return True
    
    visited.add(start)
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            if dfs(graph, neighbor, target, visited):
                return True
    
    return False

# Iterative DFS
def dfs_iterative(graph: dict, start, target) -> bool:
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node == target:
            return True
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return False
```

### BFS vs DFS Comparison

```
BFS (Queue)                      DFS (Stack/Recursion)
───────────────────              ───────────────────────

Explores level by level          Goes deep before wide

     1                                1
   / | \                            / | \
  2  3  4      ← Level 1           2  3  4
 /|  |  |\                        /|  |  |\
5 6  7  8 9    ← Level 2         5 6  7  8 9

Order: 1,2,3,4,5,6,7,8,9         Order: 1,2,5,6,3,7,4,8,9

Use for:                         Use for:
• Shortest path (unweighted)     • Path existence
• Level-order traversal          • Cycle detection
• Finding nearest                • Topological sort
                                 • Backtracking problems
```

---

## Hash-Based Search

### Hash Table Lookup

```python
def hash_search(data: dict, key) -> any:
    """
    O(1) average case lookup.
    
    Time:  O(1) average, O(n) worst (all collisions)
    Space: O(n)
    """
    return data.get(key, None)

# Building a search structure
def build_index(arr: list) -> dict:
    """Build hash map for O(1) lookup by value."""
    return {val: idx for idx, val in enumerate(arr)}

# Two Sum with hash map
def two_sum(nums: list, target: int) -> list:
    """Find indices of two numbers that sum to target."""
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:  # O(1) lookup
            return [seen[complement], i]
        seen[num] = i
    
    return []
```

---

## Interpolation Search

### For Uniformly Distributed Data

```python
def interpolation_search(arr: list, target) -> int:
    """
    Estimate position based on value distribution.
    
    Time:  O(log log n) average for uniform distribution
           O(n) worst case (non-uniform)
    Space: O(1)
    
    Best for: Large, uniformly distributed sorted arrays
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and arr[left] <= target <= arr[right]:
        if left == right:
            if arr[left] == target:
                return left
            return -1
        
        # Interpolate position
        pos = left + ((target - arr[left]) * (right - left) // 
                      (arr[right] - arr[left]))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1
```

**Intuition**: If searching for "M" in a dictionary, don't start at the middle—start near 1/3 of the way through!

---

## Classic Search Problems

### Problem 1: Find Peak Element

```python
def find_peak(nums: list) -> int:
    """
    Find a peak element (greater than neighbors).
    Binary search works because there's always a peak!
    
    Time: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid  # Peak is in left half (including mid)
        else:
            left = mid + 1  # Peak is in right half
    
    return left
```

### Problem 2: Search in 2D Matrix

```python
def search_matrix(matrix: list[list[int]], target: int) -> bool:
    """
    Search in row-wise and column-wise sorted matrix.
    Start from top-right or bottom-left.
    
    Time: O(m + n)
    """
    if not matrix:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1  # Start top-right
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1  # Go left
        else:
            row += 1  # Go down
    
    return False
```

### Problem 3: Find Minimum in Rotated Sorted Array

```python
def find_min(nums: list) -> int:
    """
    Find minimum in rotated sorted array.
    
    Time: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1  # Min is in right half
        else:
            right = mid  # Min is in left half (including mid)
    
    return nums[left]
```

---

## Search Algorithm Comparison

| Algorithm | Time (Avg) | Time (Worst) | Space | Prerequisite |
|-----------|------------|--------------|-------|--------------|
| Linear Search | O(n) | O(n) | O(1) | None |
| Binary Search | O(log n) | O(log n) | O(1) | Sorted |
| Interpolation | O(log log n) | O(n) | O(1) | Sorted + uniform |
| Hash Lookup | O(1) | O(n) | O(n) | Hash table |
| BST Search | O(log n) | O(n) | O(1) | BST structure |
| BFS/DFS | O(V+E) | O(V+E) | O(V) | Graph structure |

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                  SEARCHING ALGORITHMS                        │
├─────────────────────────────────────────────────────────────┤
│ LINEAR SEARCH: O(n)                                         │
│ • Use for: unsorted data, small datasets, linked lists      │
├─────────────────────────────────────────────────────────────┤
│ BINARY SEARCH: O(log n)                                     │
│ • Requires: SORTED array                                    │
│ • Template: left=0, right=n-1, while left<=right            │
│ • Variants: first/last occurrence, lower/upper bound        │
│ • Also: search on answer space (not just arrays)            │
├─────────────────────────────────────────────────────────────┤
│ BST SEARCH: O(log n) balanced, O(n) skewed                  │
│ • Go left if target < node, right if target > node          │
├─────────────────────────────────────────────────────────────┤
│ BFS: O(V+E) - Use queue                                     │
│ • Best for: shortest path in unweighted graph               │
│                                                             │
│ DFS: O(V+E) - Use stack/recursion                           │
│ • Best for: path existence, cycle detection                 │
├─────────────────────────────────────────────────────────────┤
│ HASH LOOKUP: O(1) average                                   │
│ • Best for: frequent lookups, when space allows             │
├─────────────────────────────────────────────────────────────┤
│ PYTHON: import bisect                                       │
│ • bisect_left(arr, x) - first position for x                │
│ • bisect_right(arr, x) - position after last x              │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Binary Search | Basic binary search | 704 |
| 2 | Search Insert Position | Lower bound | 35 |
| 3 | Find First and Last Position | Left/right bounds | 34 |
| 4 | Search in Rotated Sorted Array | Modified binary search | 33 |
| 5 | Find Peak Element | Binary search property | 162 |
| 6 | Search a 2D Matrix | 2D binary search | 74 |
| 7 | Find Minimum in Rotated Array | Rotation point | 153 |
| 8 | Sqrt(x) | Binary search on answer | 69 |
| 9 | Koko Eating Bananas | Search on answer space | 875 |
| 10 | Capacity To Ship Packages | Search on answer space | 1011 |

---

## Summary

1. **Know your data**: Sorted → binary search, unsorted → linear or hash.

2. **Binary search is powerful**: Not just for finding elements—use it on answer spaces.

3. **Learn the variations**: First/last occurrence, lower/upper bound are common interview patterns.

4. **BFS for shortest path**: In unweighted graphs, BFS guarantees shortest path.

5. **Hash for O(1)**: When you need fast lookups and can afford the space.

6. **Practice templates**: Binary search edge cases are tricky—practice until they're automatic.
