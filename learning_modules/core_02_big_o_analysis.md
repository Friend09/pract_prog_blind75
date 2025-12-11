# CORE_02: Big-O Analysis

## Concept Overview

### What Is Big-O Notation?

**Big-O notation** describes how an algorithm's resource usage (time or space) grows as the input size grows. It's a way to classify algorithms by their **worst-case scalability**.

```
┌─────────────────────────────────────────────────────────────┐
│              BIG-O IN ONE SENTENCE                          │
├─────────────────────────────────────────────────────────────┤
│  "How does the runtime grow as the input gets bigger?"      │
│                                                             │
│  Big-O ignores constants and lower-order terms to focus     │
│  on the DOMINANT factor that matters at scale.              │
└─────────────────────────────────────────────────────────────┘
```

### Why Big-O Matters for Interviews

1. **Every interview asks**: "What's the time and space complexity?"
2. **Optimization**: You need to know if your solution is good enough
3. **Tradeoffs**: Choose between time vs. space, or different algorithms
4. **Communication**: Shared vocabulary with interviewers

### The Core Idea: Growth Rate

We care about **how fast** the function grows, not its exact value:

```
Input size:     10      100       1,000       10,000      100,000
─────────────────────────────────────────────────────────────────
O(1)            1       1         1           1           1
O(log n)        3       7         10          13          17
O(n)            10      100       1,000       10,000      100,000
O(n log n)      30      700       10,000      130,000     1,700,000
O(n²)           100     10,000    1,000,000   100,000,000 10^10
O(2^n)          1,024   10^30     10^301      ...         ...

↑ Small inputs: differences are minimal
↓ Large inputs: differences are MASSIVE
```

---

## Asymptotic Notation Family

### Big-O (O): Upper Bound

**Definition**: f(n) = O(g(n)) means f(n) grows **at most as fast** as g(n).

```
T(n) = 3n² + 5n + 100

As n → ∞:
- 3n² dominates 5n (n² >> n for large n)
- 3n² dominates 100 (n² >> constant)
- The coefficient 3 doesn't change the growth rate

Therefore: T(n) = O(n²)
```

**What Big-O ignores**:
- Constants: O(3n) = O(n)
- Lower-order terms: O(n² + n) = O(n²)

### Big-Ω (Omega): Lower Bound

**Definition**: f(n) = Ω(g(n)) means f(n) grows **at least as fast** as g(n).

```
Example: Any comparison-based sort is Ω(n log n)
(You can't do better than n log n comparisons)
```

### Big-Θ (Theta): Tight Bound

**Definition**: f(n) = Θ(g(n)) means f(n) grows **exactly as fast** as g(n).

```
Example: Merge sort is Θ(n log n)
(It's both O(n log n) and Ω(n log n))
```

### What Interviewers Usually Want

In interviews, "Big-O" typically means the **worst-case upper bound**. When asked "What's the complexity?", give:
- **Time complexity**: Operations performed
- **Space complexity**: Memory used (beyond input)

---

## Common Complexity Classes

### Visual Comparison

```
                              │
Time                          │                    O(2^n)
 ↑                            │                   /
 │                            │                  /
 │                            │                 /     O(n²)
 │                            │                /    ╱
 │                            │               /   ╱
 │                            │              /  ╱     O(n log n)
 │                            │             / ╱    ╱──
 │                            │            /╱   ╱──
 │                            │           ╱  ╱──      O(n)
 │                            │         ╱╱──       ╱──────
 │                            │       ╱──     ╱────────
 │                            │    ╱──   ╱──────           O(log n)
 │                            │ ╱────────────────────────────────
 │                            │══════════════════════════════════ O(1)
 └────────────────────────────┴──────────────────────────────────→
                              │                            Input size (n)
```

### Complexity Classes Explained

| Complexity | Name | Example | Intuition |
|------------|------|---------|-----------|
| O(1) | Constant | Array access, hash lookup | Same time regardless of input size |
| O(log n) | Logarithmic | Binary search | Halving the problem each step |
| O(n) | Linear | Single loop through array | Touch each element once |
| O(n log n) | Linearithmic | Merge sort, heap sort | Divide & conquer with linear merge |
| O(n²) | Quadratic | Nested loops, bubble sort | Compare every pair |
| O(n³) | Cubic | Triple nested loops, matrix multiplication | 3D iteration |
| O(2^n) | Exponential | Subsets, naive fibonacci | Double with each added element |
| O(n!) | Factorial | Permutations | All orderings |

---

## Detailed Examples of Each Complexity

### O(1) - Constant Time

**Definition**: Runtime doesn't depend on input size.

```python
def get_first(arr):
    """O(1) - Always one operation"""
    return arr[0] if arr else None

def hash_lookup(dictionary, key):
    """O(1) average - Hash table magic"""
    return dictionary.get(key)

def is_even(n):
    """O(1) - Single operation"""
    return n % 2 == 0

def swap(arr, i, j):
    """O(1) - Three operations, but constant"""
    arr[i], arr[j] = arr[j], arr[i]
```

**Key insight**: Even if you do 1000 operations, if it's always 1000 regardless of n, it's O(1).

---

### O(log n) - Logarithmic Time

**Definition**: Runtime grows slowly—doubling input adds just one more step.

```python
def binary_search(arr, target):
    """
    O(log n) - Halve search space each iteration
    
    Why log n?
    n=16: 16→8→4→2→1 = 4 steps = log₂(16)
    n=32: 32→16→8→4→2→1 = 5 steps = log₂(32)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

**Logarithm intuition**:
```
log₂(n) answers: "How many times can I divide n by 2 until I reach 1?"

n=1:    0 divisions → log₂(1) = 0
n=2:    1 division  → log₂(2) = 1
n=4:    2 divisions → log₂(4) = 2
n=8:    3 divisions → log₂(8) = 3
n=1024: 10 divisions → log₂(1024) = 10
n=1M:   20 divisions → log₂(1M) ≈ 20
```

---

### O(n) - Linear Time

**Definition**: Runtime grows proportionally with input size.

```python
def find_max(arr):
    """O(n) - Must check every element"""
    if not arr:
        return None
    
    maximum = arr[0]
    for num in arr:        # n iterations
        if num > maximum:  # O(1) per iteration
            maximum = num
    return maximum

def sum_array(arr):
    """O(n) - One pass through array"""
    total = 0
    for num in arr:
        total += num
    return total

def contains_duplicate(arr):
    """O(n) - Hash set approach"""
    seen = set()
    for num in arr:
        if num in seen:    # O(1) hash lookup
            return True
        seen.add(num)      # O(1) hash insert
    return False
```

**Key insight**: A single loop through n elements = O(n).

---

### O(n log n) - Linearithmic Time

**Definition**: Slightly worse than linear, common in efficient sorting.

```python
def merge_sort(arr):
    """
    O(n log n) - Divide and conquer
    
    Why n log n?
    - log n levels of recursion (dividing in half)
    - O(n) work at each level (merging)
    - Total: O(n) × O(log n) = O(n log n)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])     # T(n/2)
    right = merge_sort(arr[mid:])    # T(n/2)
    return merge(left, right)        # O(n)

def merge(left, right):
    """O(n) - Linear merge of two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**Visualization**:
```
Level 0:  [8 elements]                    → n work
Level 1:  [4 elements] [4 elements]       → n work
Level 2:  [2][2] [2][2]                   → n work
Level 3:  [1][1][1][1][1][1][1][1]        → n work
          ↑
          log₂(8) = 3 levels + 1

Total: n work × log(n) levels = O(n log n)
```

---

### O(n²) - Quadratic Time

**Definition**: Runtime grows with the square of input size.

```python
def bubble_sort(arr):
    """
    O(n²) - Nested loops, each up to n iterations
    """
    n = len(arr)
    for i in range(n):           # n iterations
        for j in range(n - 1):   # n-1 iterations
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def find_pairs_with_sum(arr, target):
    """O(n²) - Check all pairs"""
    pairs = []
    n = len(arr)
    
    for i in range(n):           # n iterations
        for j in range(i + 1, n):  # ~n/2 iterations on average
            if arr[i] + arr[j] == target:
                pairs.append((arr[i], arr[j]))
    
    return pairs
    # n × n/2 = n²/2 = O(n²)
```

**Warning sign**: Nested loops over the same input.

---

### O(2^n) - Exponential Time

**Definition**: Runtime doubles with each additional input element.

```python
def fibonacci_naive(n):
    """
    O(2^n) - Each call spawns two more calls
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

def subsets(nums):
    """
    O(2^n) - Generate all 2^n subsets
    Each element: include or exclude (2 choices)
    n elements: 2 × 2 × ... × 2 = 2^n subsets
    """
    result = []
    
    def backtrack(index, current):
        if index == len(nums):
            result.append(current.copy())
            return
        
        # Exclude
        backtrack(index + 1, current)
        
        # Include
        current.append(nums[index])
        backtrack(index + 1, current)
        current.pop()
    
    backtrack(0, [])
    return result
```

**Growth illustration**:
```
n=10:   2^10  = 1,024
n=20:   2^20  = 1,048,576 (1 million)
n=30:   2^30  = 1,073,741,824 (1 billion)
n=40:   2^40  = ~1 trillion

This gets VERY slow VERY fast!
```

---

### O(n!) - Factorial Time

**Definition**: All possible orderings. Grows faster than exponential.

```python
def permutations(arr):
    """
    O(n!) - Generate all n! permutations
    
    n=3: 3! = 6 permutations
    n=5: 5! = 120 permutations
    n=10: 10! = 3,628,800 permutations
    """
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for perm in permutations(rest):
            result.append([arr[i]] + perm)
    
    return result
```

**Why n!?**
```
First position:  n choices
Second position: n-1 choices
Third position:  n-2 choices
...
Total: n × (n-1) × (n-2) × ... × 1 = n!
```

---

## How to Analyze Code

### Step-by-Step Method

```
1. IDENTIFY the operations that scale with input
2. COUNT how many times each runs
3. MULTIPLY nested operations
4. ADD sequential operations
5. DROP constants and lower-order terms
6. IDENTIFY the dominant term
```

### Example Analysis: Nested Loops

```python
def example(arr):
    n = len(arr)
    count = 0
    
    for i in range(n):            # Runs n times
        for j in range(n):        # Runs n times (for each i)
            count += 1            # O(1) operation
    
    for k in range(n):            # Runs n times
        count += 1                # O(1) operation
    
    return count

# Analysis:
# First nested loop: n × n × O(1) = O(n²)
# Second loop: n × O(1) = O(n)
# Total: O(n²) + O(n) = O(n²)  [n² dominates]
```

### Example Analysis: Different Loop Bounds

```python
def example2(arr):
    n = len(arr)
    count = 0
    
    for i in range(n):            # n iterations
        for j in range(i):        # 0, 1, 2, ..., n-1 iterations
            count += 1
    
    return count

# Analysis:
# When i=0: inner loop runs 0 times
# When i=1: inner loop runs 1 time
# When i=2: inner loop runs 2 times
# ...
# When i=n-1: inner loop runs n-1 times
#
# Total: 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = O(n²)
```

### Example Analysis: Logarithmic Pattern

```python
def example3(n):
    count = 0
    i = n
    
    while i > 0:
        count += 1
        i = i // 2  # Halving each iteration
    
    return count

# Analysis:
# n → n/2 → n/4 → n/8 → ... → 1
# Number of iterations = log₂(n)
# Time: O(log n)
```

### Example Analysis: Multiply When Nested

```python
def example4(n):
    count = 0
    i = n
    
    while i > 0:           # log n iterations
        for j in range(n):  # n iterations each time
            count += 1
        i = i // 2
    
    return count

# Analysis:
# Outer loop: O(log n) iterations
# Inner loop: O(n) iterations per outer iteration
# Total: O(log n) × O(n) = O(n log n)
```

---

## Space Complexity

### What Counts as Space?

**Auxiliary space**: Extra space used by the algorithm (excluding input).

| Category | Examples | Notes |
|----------|----------|-------|
| Variables | `i`, `count`, `sum` | O(1) |
| Data structures | Arrays, hash maps, stacks | O(size) |
| Recursion stack | Function call frames | O(depth) |
| Output | Result array | Sometimes counted, sometimes not |

### Common Space Patterns

```python
# O(1) Space - Fixed variables
def find_max(arr):
    max_val = arr[0]       # One variable
    for num in arr:
        max_val = max(max_val, num)
    return max_val

# O(n) Space - Linear data structure
def two_sum(arr, target):
    seen = {}              # Hash map grows with input
    for i, num in enumerate(arr):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []

# O(n) Space - Recursion stack
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    # Stack depth: n

# O(log n) Space - Balanced recursion
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
    # Stack depth: log n
```

---

## Time-Space Tradeoffs

### Classic Example: Two Sum

```python
# Approach 1: O(n²) time, O(1) space
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

# Approach 2: O(n) time, O(n) space
def two_sum_hash(nums, target):
    seen = {}  # Trade space for time
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### Decision Framework

| Constraint | Preference |
|------------|------------|
| Memory-limited system | Prioritize space |
| Time-critical application | Prioritize time |
| Interview default | Usually prioritize time |
| Can preprocess once, query many | Space is OK for preprocessing |

---

## Amortized Analysis

### What Is Amortized Complexity?

**Amortized** complexity is the average time per operation over a sequence of operations, even if individual operations vary.

### Dynamic Array Example

```python
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.arr = [None] * self.capacity
    
    def append(self, item):
        """
        Usually O(1), but O(n) when resizing
        Amortized: O(1)
        """
        if self.size == self.capacity:
            # Double the capacity - O(n) copy
            self.capacity *= 2
            new_arr = [None] * self.capacity
            for i in range(self.size):
                new_arr[i] = self.arr[i]
            self.arr = new_arr
        
        self.arr[self.size] = item
        self.size += 1
```

**Why amortized O(1)?**
```
Insert n elements:
- Most inserts: O(1)
- Resizes at: 1, 2, 4, 8, 16, ..., n elements
- Total copy operations: 1 + 2 + 4 + 8 + ... + n ≈ 2n

Average per operation: 2n / n = O(1)
```

---

## Best, Average, Worst Case

### Quick Sort Example

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[0]  # Choosing first element as pivot
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    
    return quicksort(left) + [pivot] + quicksort(right)
```

| Case | When It Occurs | Complexity |
|------|----------------|------------|
| **Best** | Pivot always splits evenly | O(n log n) |
| **Average** | Random pivots | O(n log n) |
| **Worst** | Already sorted, pivot is min/max | O(n²) |

**What interviewers want**: Usually worst case, unless they ask for average.

---

## Common Complexity Derivations

### 1. Arithmetic Series

```
1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²)

Used in:
- Nested loops where inner loop depends on outer
- Bubble sort, selection sort, insertion sort
```

### 2. Geometric Series

```
1 + 2 + 4 + 8 + ... + n = 2n - 1 = O(n)

Used in:
- Dynamic array resizing
- Divide and conquer auxiliary space
```

### 3. Harmonic Series

```
1 + 1/2 + 1/3 + ... + 1/n = O(log n)

Used in:
- Sieve of Eratosthenes analysis
- Some skip list operations
```

### 4. Master Theorem (Quick Reference)

For recurrences of form: T(n) = aT(n/b) + O(n^d)

```
If log_b(a) < d:  T(n) = O(n^d)
If log_b(a) = d:  T(n) = O(n^d log n)
If log_b(a) > d:  T(n) = O(n^(log_b(a)))
```

**Examples**:
```
T(n) = 2T(n/2) + O(n)     → a=2, b=2, d=1
log_2(2) = 1 = d          → O(n log n)    [Merge Sort]

T(n) = T(n/2) + O(1)      → a=1, b=2, d=0
log_2(1) = 0 = d          → O(log n)      [Binary Search]

T(n) = 2T(n/2) + O(1)     → a=2, b=2, d=0
log_2(2) = 1 > 0 = d      → O(n)          [Binary Tree Traversal]
```

---

## Quick Complexity Reference

### Data Structure Operations

| Structure | Access | Search | Insert | Delete | Space |
|-----------|--------|--------|--------|--------|-------|
| Array | O(1) | O(n) | O(n) | O(n) | O(n) |
| Dynamic Array | O(1) | O(n) | O(1)* | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) | O(n) |
| Hash Table | - | O(1)* | O(1)* | O(1)* | O(n) |
| BST (balanced) | - | O(log n) | O(log n) | O(log n) | O(n) |
| Heap | - | O(n) | O(log n) | O(log n) | O(n) |

*Amortized or average case

### Sorting Algorithms

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) |

---

## Interview Tips

### How to Communicate Complexity

```
GOOD: "This solution is O(n) time and O(1) space because we make a 
      single pass through the array using only a few variables."

BAD:  "This is O(n)."  [Too brief, no explanation]

GOOD: "The worst case is O(n²) when the array is already sorted and 
      we always pick the minimum as pivot, but average case is 
      O(n log n) with random pivot selection."

BAD:  "It's sometimes O(n²)."  [Vague, no analysis]
```

### Common Follow-Up Questions

| Question | What They're Testing |
|----------|---------------------|
| "Can you do better?" | Know optimal complexities for common problems |
| "What about space?" | Always mention both time AND space |
| "What's the worst case?" | Understand when bad cases occur |
| "What if the input is sorted?" | Know how properties affect complexity |

### Red Flags

- Saying O(n) for clearly nested loops
- Forgetting recursion space complexity
- Not mentioning amortized when relevant
- Confusing log bases (they're all equivalent in Big-O: O(log₂ n) = O(log₁₀ n) = O(ln n))

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    BIG-O ANALYSIS                           │
├─────────────────────────────────────────────────────────────┤
│ COMMON COMPLEXITIES (fastest to slowest):                   │
│   O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n)     │
├─────────────────────────────────────────────────────────────┤
│ RULES:                                                      │
│ • Drop constants: O(2n) = O(n)                              │
│ • Drop lower terms: O(n² + n) = O(n²)                       │
│ • Nested loops: multiply                                    │
│ • Sequential loops: add (then drop lower)                   │
│ • Recursive calls: count nodes × work per node              │
├─────────────────────────────────────────────────────────────┤
│ QUICK PATTERNS:                                             │
│ • Single loop → O(n)                                        │
│ • Nested loops (same range) → O(n²)                         │
│ • Halving each step → O(log n)                              │
│ • All subsets → O(2^n)                                      │
│ • All permutations → O(n!)                                  │
├─────────────────────────────────────────────────────────────┤
│ SPACE COMPLEXITY:                                           │
│ • Fixed variables → O(1)                                    │
│ • Array of size n → O(n)                                    │
│ • Recursion → O(depth)                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Analyze These Code Snippets

```python
# Problem 1: What's the time complexity?
def mystery1(n):
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                count += 1
    return count
# Answer: O(n³)

# Problem 2: What's the time complexity?
def mystery2(n):
    count = 0
    i = 1
    while i < n:
        count += 1
        i *= 2
    return count
# Answer: O(log n) - i doubles each time

# Problem 3: What's the time complexity?
def mystery3(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            pass
# Answer: O(n²) - n + (n-1) + (n-2) + ... + 1 = n(n+1)/2

# Problem 4: What's the time and space?
def mystery4(n):
    if n <= 0:
        return 0
    return n + mystery4(n - 1)
# Answer: Time O(n), Space O(n) [call stack]
```

---

## Summary: Key Takeaways

1. **Focus on growth rate**: Big-O describes how runtime scales, not exact time.

2. **Worst case by default**: Unless asked, give worst-case complexity.

3. **Always mention both**: Time AND space complexity.

4. **Know the hierarchy**: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!)

5. **Explain your reasoning**: Don't just state the answer—show how you derived it.

6. **Watch for recursion**: Remember to count stack space for recursive solutions.
