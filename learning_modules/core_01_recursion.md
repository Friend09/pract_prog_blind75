# CORE_01: Recursion

## Concept Overview

### What Is Recursion?

**Recursion** is a problem-solving technique where a function calls itself to solve smaller instances of the same problem. It's not just a programming trick—it's a fundamental way of thinking about problems that maps naturally to many algorithmic patterns.

```
┌─────────────────────────────────────────────────────────────┐
│                    RECURSION IN ONE SENTENCE                │
├─────────────────────────────────────────────────────────────┤
│  "To solve a big problem, solve smaller versions of the    │
│   same problem, then combine the results."                  │
└─────────────────────────────────────────────────────────────┘
```

### Why Recursion Matters for Interviews

Recursion is the **foundation** of:
- **Tree traversals** (inorder, preorder, postorder)
- **Graph algorithms** (DFS, connected components)
- **Backtracking** (permutations, combinations, N-Queens)
- **Divide and conquer** (merge sort, quick sort)
- **Dynamic programming** (recursive + memoization)

If you don't deeply understand recursion, these patterns will feel like magic. With recursion mastery, they become intuitive.

### The Two Essential Components

Every recursive function needs exactly two things:

```python
def recursive_function(problem):
    # 1. BASE CASE: When to stop recursing
    if problem_is_simple_enough:
        return simple_answer
    
    # 2. RECURSIVE CASE: Break down and recurse
    smaller_problem = make_problem_smaller(problem)
    result = recursive_function(smaller_problem)
    return combine_results(result)
```

| Component | Purpose | What Happens Without It |
|-----------|---------|-------------------------|
| **Base Case** | Stops recursion | Infinite loop → Stack overflow |
| **Recursive Case** | Makes progress toward base case | Never reaches solution |

---

## The Call Stack: How Recursion Actually Works

### What Is the Call Stack?

When a function calls itself (or any function), the computer needs to remember:
- Where to return after the call completes
- The values of local variables
- The parameters passed to the function

This information is stored in a **stack frame** on the **call stack**.

```
┌─────────────────────────────────────────────────────────────┐
│                      THE CALL STACK                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Each recursive call adds a new frame:                     │
│                                                             │
│   ┌──────────────────┐                                      │
│   │ factorial(1)     │  ← Top (most recent call)            │
│   │ n=1, return 1    │                                      │
│   ├──────────────────┤                                      │
│   │ factorial(2)     │                                      │
│   │ n=2, waiting...  │                                      │
│   ├──────────────────┤                                      │
│   │ factorial(3)     │                                      │
│   │ n=3, waiting...  │                                      │
│   ├──────────────────┤                                      │
│   │ factorial(4)     │  ← Bottom (original call)            │
│   │ n=4, waiting...  │                                      │
│   └──────────────────┘                                      │
│                                                             │
│   Stack grows DOWN as we make more calls                    │
│   Results flow UP as calls return                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Concrete Example: Factorial

```python
def factorial(n):
    """
    Calculate n! = n × (n-1) × (n-2) × ... × 1
    
    Base case: 0! = 1 (by definition)
    Recursive case: n! = n × (n-1)!
    """
    # Base case
    if n == 0:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)
```

### Step-by-Step Execution Trace

**Input**: `factorial(4)`

```
PHASE 1: WINDING (Building up the stack)
═══════════════════════════════════════

Call 1: factorial(4)
        n=4, not base case
        Need: 4 * factorial(3)
        Status: WAITING
        
        │ Call 2: factorial(3)
        │         n=3, not base case
        │         Need: 3 * factorial(2)
        │         Status: WAITING
        │         
        │         │ Call 3: factorial(2)
        │         │         n=2, not base case
        │         │         Need: 2 * factorial(1)
        │         │         Status: WAITING
        │         │         
        │         │         │ Call 4: factorial(1)
        │         │         │         n=1, not base case
        │         │         │         Need: 1 * factorial(0)
        │         │         │         Status: WAITING
        │         │         │         
        │         │         │         │ Call 5: factorial(0)
        │         │         │         │         n=0, BASE CASE!
        │         │         │         │         Return: 1
        │         │         │         │
```

```
PHASE 2: UNWINDING (Popping the stack, computing results)
═════════════════════════════════════════════════════════

        │         │         │         │ factorial(0) returns 1
        │         │         │         ↓
        │         │         │ factorial(1) computes 1 * 1 = 1, returns 1
        │         │         ↓
        │         │ factorial(2) computes 2 * 1 = 2, returns 2
        │         ↓
        │ factorial(3) computes 3 * 2 = 6, returns 6
        ↓
factorial(4) computes 4 * 6 = 24, returns 24

FINAL RESULT: 24
```

### Stack Frame Visualization

| Step | Action | Stack Contents | Return Value |
|------|--------|----------------|--------------|
| 1 | Call factorial(4) | `[f(4)]` | - |
| 2 | Call factorial(3) | `[f(4), f(3)]` | - |
| 3 | Call factorial(2) | `[f(4), f(3), f(2)]` | - |
| 4 | Call factorial(1) | `[f(4), f(3), f(2), f(1)]` | - |
| 5 | Call factorial(0) | `[f(4), f(3), f(2), f(1), f(0)]` | - |
| 6 | Return from f(0) | `[f(4), f(3), f(2), f(1)]` | 1 |
| 7 | Return from f(1) | `[f(4), f(3), f(2)]` | 1 |
| 8 | Return from f(2) | `[f(4), f(3)]` | 2 |
| 9 | Return from f(3) | `[f(4)]` | 6 |
| 10 | Return from f(4) | `[]` | **24** |

---

## Recursion Tree: Visualizing the Calls

For problems with multiple recursive calls, we draw a **recursion tree**.

### Example: Fibonacci Numbers

```python
def fibonacci(n):
    """
    F(0) = 0, F(1) = 1
    F(n) = F(n-1) + F(n-2)
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### Recursion Tree for fibonacci(5)

```
                         fib(5)
                        /      \
                    fib(4)      fib(3)
                   /     \      /     \
               fib(3)  fib(2) fib(2) fib(1)
               /    \   /   \   /   \    |
           fib(2) fib(1) fib(1) fib(0) fib(1) fib(0)  1
           /   \    |      |      |      |      |
       fib(1) fib(0) 1     1      0      1      0
          |     |
          1     0

Node count: 15 nodes for fib(5)
Pattern: O(2^n) nodes - exponential!
```

### Why This Tree Matters

1. **Visualize work**: Each node = one function call
2. **Identify redundancy**: fib(3) computed twice, fib(2) computed 3 times
3. **Understand complexity**: Tree size = time complexity
4. **Motivate optimization**: Repeated work → memoization (DP)

---

## Types of Recursion

### 1. Linear Recursion (One Recursive Call)

```python
def sum_array(arr, index=0):
    """Sum elements: one call per element."""
    if index == len(arr):
        return 0
    return arr[index] + sum_array(arr, index + 1)
```

**Recursion pattern**: Single chain
```
sum([1,2,3,4]) → sum([2,3,4]) → sum([3,4]) → sum([4]) → sum([]) = 0
```

**Time**: O(n), **Space**: O(n) stack frames

---

### 2. Binary Recursion (Two Recursive Calls)

```python
def binary_tree_sum(node):
    """Sum all values in binary tree."""
    if node is None:
        return 0
    return node.val + binary_tree_sum(node.left) + binary_tree_sum(node.right)
```

**Recursion pattern**: Binary tree
```
        sum(root)
       /         \
  sum(left)    sum(right)
   /    \        /    \
  ...   ...    ...    ...
```

**Time**: O(n) nodes, **Space**: O(h) where h = height

---

### 3. Multiple Recursion (Many Recursive Calls)

```python
def permutations(arr, start=0):
    """Generate all permutations."""
    if start == len(arr):
        print(arr)
        return
    
    for i in range(start, len(arr)):
        arr[start], arr[i] = arr[i], arr[start]  # Swap
        permutations(arr, start + 1)              # Recurse
        arr[start], arr[i] = arr[i], arr[start]  # Backtrack
```

**Recursion pattern**: n-ary tree (branching factor varies)

**Time**: O(n!), **Space**: O(n)

---

### 4. Tail Recursion

A function is **tail recursive** if the recursive call is the **last operation**.

```python
# NOT tail recursive (multiplication happens AFTER recursive call returns)
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)  # Must wait for result, then multiply

# Tail recursive (recursive call IS the return, nothing after)
def factorial_tail(n, accumulator=1):
    if n == 0:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # Just return the call
```

**Why it matters**: Tail recursion can be optimized by compilers to use O(1) space (tail call optimization). Python doesn't do this, but many languages do.

---

## Common Recursion Patterns

### Pattern 1: Process and Recurse

Process current element, then recurse on the rest.

```python
def print_list(lst, index=0):
    if index == len(lst):
        return
    print(lst[index])           # Process current
    print_list(lst, index + 1)  # Recurse on rest
```

### Pattern 2: Recurse and Combine

Recurse first, then combine results.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # Recurse
    right = merge_sort(arr[mid:])   # Recurse
    return merge(left, right)       # Combine
```

### Pattern 3: Choose, Explore, Unchoose (Backtracking)

Make a choice, recurse, then undo the choice.

```python
def subsets(nums):
    result = []
    
    def backtrack(index, current):
        result.append(current.copy())
        
        for i in range(index, len(nums)):
            current.append(nums[i])      # Choose
            backtrack(i + 1, current)    # Explore
            current.pop()                # Unchoose
    
    backtrack(0, [])
    return result
```

### Pattern 4: Divide and Conquer

Split problem, solve halves, combine.

```python
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)   # Left half
    else:
        return binary_search(arr, target, mid + 1, right)  # Right half
```

---

## Recursion vs Iteration

### Every Recursion Can Be Converted to Iteration

Recursion uses the **call stack** implicitly. Iteration can use an **explicit stack**.

```python
# Recursive
def factorial_recursive(n):
    if n == 0:
        return 1
    return n * factorial_recursive(n - 1)

# Iterative equivalent
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

### When to Use Which?

| Factor | Recursion | Iteration |
|--------|-----------|-----------|
| **Readability** | Often cleaner for trees, graphs | Cleaner for simple loops |
| **Space** | O(depth) stack frames | O(1) if no explicit stack |
| **Performance** | Function call overhead | Slightly faster |
| **Natural fit** | Tree/graph traversal, divide & conquer | Linear processing |
| **Interview** | Expected for trees, backtracking | Fine for simple problems |

### Converting Tree Traversal: Recursive to Iterative

```python
# Recursive (clean, natural)
def inorder_recursive(root):
    if not root:
        return []
    return inorder_recursive(root.left) + [root.val] + inorder_recursive(root.right)

# Iterative (explicit stack)
def inorder_iterative(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go left as far as possible
        while current:
            stack.append(current)
            current = current.left
        
        # Process node
        current = stack.pop()
        result.append(current.val)
        
        # Go right
        current = current.right
    
    return result
```

---

## Complexity Analysis for Recursion

### Time Complexity: Count the Nodes

**Time = (Number of recursive calls) × (Work per call)**

```
For factorial(n):
- Number of calls: n + 1
- Work per call: O(1)
- Total: O(n)

For fibonacci(n) [naive]:
- Number of calls: O(2^n) (tree nodes)
- Work per call: O(1)
- Total: O(2^n)

For merge_sort(arr):
- Number of levels: O(log n)
- Work per level: O(n)
- Total: O(n log n)
```

### Space Complexity: Stack Depth

**Space = (Maximum recursion depth) × (Space per frame)**

```
For factorial(n):
- Max depth: n
- Space per frame: O(1)
- Total: O(n)

For binary_search(arr):
- Max depth: O(log n)
- Space per frame: O(1)
- Total: O(log n)

For tree traversal:
- Max depth: O(h) where h = tree height
- Balanced tree: O(log n)
- Skewed tree: O(n)
```

### The Recurrence Relation Method

Express time complexity as a recurrence, then solve.

```
Factorial:
T(n) = T(n-1) + O(1)
T(0) = O(1)
Solution: T(n) = O(n)

Fibonacci (naive):
T(n) = T(n-1) + T(n-2) + O(1)
T(0) = T(1) = O(1)
Solution: T(n) = O(2^n) [approximately O(φ^n) where φ ≈ 1.618]

Merge Sort:
T(n) = 2T(n/2) + O(n)
T(1) = O(1)
Solution: T(n) = O(n log n) [Master theorem]
```

---

## Common Mistakes and How to Avoid Them

### Mistake 1: Missing or Wrong Base Case

```python
# BUG: No base case → infinite recursion
def countdown(n):
    print(n)
    countdown(n - 1)  # Never stops!

# FIXED: Add base case
def countdown(n):
    if n < 0:         # Base case
        return
    print(n)
    countdown(n - 1)
```

### Mistake 2: Not Making Progress Toward Base Case

```python
# BUG: n never decreases
def broken(n):
    if n == 0:
        return
    broken(n)  # Same n forever!

# FIXED: Decrease n
def working(n):
    if n == 0:
        return
    working(n - 1)  # Progress toward base case
```

### Mistake 3: Modifying Shared State Incorrectly

```python
# BUG: result is shared and modified
def broken_subsets(nums):
    result = []
    current = []
    
    def backtrack(index):
        result.append(current)  # Appending reference!
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1)
            current.pop()
    
    backtrack(0)
    return result  # All sublists point to same empty list!

# FIXED: Copy the current state
def working_subsets(nums):
    result = []
    current = []
    
    def backtrack(index):
        result.append(current.copy())  # Copy!
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1)
            current.pop()
    
    backtrack(0)
    return result
```

### Mistake 4: Stack Overflow on Large Inputs

```python
# BUG: Python's default recursion limit is ~1000
def sum_to_n(n):
    if n == 0:
        return 0
    return n + sum_to_n(n - 1)

sum_to_n(10000)  # RecursionError!

# SOLUTION 1: Increase limit (use carefully)
import sys
sys.setrecursionlimit(20000)

# SOLUTION 2: Convert to iteration
def sum_to_n_iterative(n):
    return n * (n + 1) // 2  # Or use a loop
```

---

## Classic Recursive Problems

### Problem 1: Reverse a String

```python
def reverse_string(s: str) -> str:
    """
    Time: O(n), Space: O(n)
    """
    # Base case
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])

# Trace: reverse("hello")
# = 'o' + reverse("hell")
# = 'o' + 'l' + reverse("hel")
# = 'o' + 'l' + 'l' + reverse("he")
# = 'o' + 'l' + 'l' + 'e' + reverse("h")
# = 'o' + 'l' + 'l' + 'e' + 'h'
# = "olleh"
```

### Problem 2: Check Palindrome

```python
def is_palindrome(s: str) -> bool:
    """
    Time: O(n), Space: O(n)
    """
    # Base cases
    if len(s) <= 1:
        return True
    
    # Check first and last, then recurse on middle
    if s[0] != s[-1]:
        return False
    
    return is_palindrome(s[1:-1])
```

### Problem 3: Power Function

```python
def power(base: float, exp: int) -> float:
    """
    Calculate base^exp efficiently.
    Time: O(log n), Space: O(log n)
    """
    # Base cases
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    
    # Divide and conquer
    half = power(base, exp // 2)
    
    if exp % 2 == 0:
        return half * half
    else:
        return half * half * base

# Why O(log n)?
# power(2, 10) → power(2, 5) → power(2, 2) → power(2, 1) → power(2, 0)
# Only log₂(n) calls!
```

### Problem 4: Flatten Nested List

```python
def flatten(nested: list) -> list:
    """
    Flatten arbitrarily nested lists.
    Time: O(total elements), Space: O(max depth)
    """
    result = []
    
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))  # Recurse on nested list
        else:
            result.append(item)
    
    return result

# Example:
# flatten([1, [2, 3], [4, [5, 6]]]) → [1, 2, 3, 4, 5, 6]
```

### Problem 5: Generate All Subsets

```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all 2^n subsets.
    Time: O(n × 2^n), Space: O(n)
    """
    result = []
    
    def backtrack(index: int, current: list[int]):
        # Every state is a valid subset
        result.append(current.copy())
        
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result
```

---

## Recursion in Tree and Graph Problems

### Tree Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Preorder: Root → Left → Right
def preorder(node):
    if not node:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)

# Inorder: Left → Root → Right
def inorder(node):
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

# Postorder: Left → Right → Root
def postorder(node):
    if not node:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]
```

### Tree Height

```python
def tree_height(node) -> int:
    """
    Height = max distance from root to any leaf.
    Time: O(n), Space: O(h)
    """
    if not node:
        return 0
    
    left_height = tree_height(node.left)
    right_height = tree_height(node.right)
    
    return 1 + max(left_height, right_height)
```

### Graph DFS

```python
def dfs(graph: dict, node: str, visited: set = None) -> list:
    """
    Depth-first traversal of graph.
    Time: O(V + E), Space: O(V)
    """
    if visited is None:
        visited = set()
    
    if node in visited:
        return []
    
    visited.add(node)
    result = [node]
    
    for neighbor in graph.get(node, []):
        result.extend(dfs(graph, neighbor, visited))
    
    return result
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                       RECURSION                             │
├─────────────────────────────────────────────────────────────┤
│ ESSENTIAL COMPONENTS:                                       │
│ 1. Base Case: When to stop                                  │
│ 2. Recursive Case: Call self with smaller problem           │
├─────────────────────────────────────────────────────────────┤
│ MENTAL MODEL:                                               │
│ • Trust the recursion (assume recursive call works)         │
│ • Focus on ONE level at a time                              │
│ • Ensure progress toward base case                          │
├─────────────────────────────────────────────────────────────┤
│ COMPLEXITY:                                                 │
│ Time:  Count nodes in recursion tree × work per node        │
│ Space: Maximum depth of recursion × space per frame         │
├─────────────────────────────────────────────────────────────┤
│ COMMON PATTERNS:                                            │
│ • Linear: f(n) calls f(n-1) once                            │
│ • Binary: f(n) calls f(n/2) twice (trees)                   │
│ • Multiple: f(n) calls f(n-1) many times (backtracking)     │
├─────────────────────────────────────────────────────────────┤
│ WATCH OUT FOR:                                              │
│ • Missing base case → infinite recursion                    │
│ • Not making progress → infinite recursion                  │
│ • Modifying shared state → wrong answers                    │
│ • Deep recursion → stack overflow                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Beginner
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Fibonacci Number | Basic recursion | 509 |
| 2 | Reverse String | Process and recurse | 344 |
| 3 | Merge Two Sorted Lists | Recursive merge | 21 |

### Intermediate
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Maximum Depth of Binary Tree | Tree recursion | 104 |
| 2 | Same Tree | Compare recursively | 100 |
| 3 | Symmetric Tree | Mirror recursion | 101 |
| 4 | Invert Binary Tree | Transform recursively | 226 |
| 5 | Power of Two | Divide by 2 | 231 |

### Advanced
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Validate BST | Pass constraints down | 98 |
| 2 | Flatten Binary Tree | Complex recursion | 114 |
| 3 | Lowest Common Ancestor | Return values up | 236 |

---

## Summary: Key Takeaways

1. **Two components**: Every recursion needs a base case and a recursive case.

2. **Trust the recursion**: Focus on one level—assume the recursive call does its job correctly.

3. **The call stack**: Understand how function calls are stacked and unwound.

4. **Recursion tree**: Visualize multiple calls to understand complexity and identify optimization opportunities.

5. **Space matters**: Recursion uses O(depth) space for the call stack.

6. **Foundation for patterns**: Mastering recursion unlocks trees, graphs, backtracking, divide & conquer, and DP.
