# CORE_04: Sorting Algorithms

## Overview

### Why Sorting Matters

Sorting is one of the most fundamental operations in computer science. Many algorithms require sorted data to work efficiently:

| Prerequisite Sorting Enables | Example |
|------------------------------|---------|
| Binary Search | O(log n) search vs O(n) |
| Two Pointers | Find pairs efficiently |
| Duplicate Detection | Adjacent duplicates |
| Merge Operations | Merge K sorted lists |
| Interval Problems | Process in order |

### Sorting Algorithm Categories

```
┌─────────────────────────────────────────────────────────────┐
│                SORTING ALGORITHM TAXONOMY                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  COMPARISON-BASED                NON-COMPARISON              │
│  (Lower bound: Ω(n log n))       (Can beat Ω(n log n))      │
│                                                             │
│  ├── Simple: O(n²)               ├── Counting Sort: O(n+k) │
│  │   ├── Bubble Sort             ├── Radix Sort: O(d(n+k)) │
│  │   ├── Selection Sort          └── Bucket Sort: O(n+k)   │
│  │   └── Insertion Sort                                    │
│  │                                                         │
│  └── Efficient: O(n log n)                                 │
│      ├── Merge Sort (stable)                               │
│      ├── Quick Sort (in-place)                             │
│      └── Heap Sort (in-place)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison-Based Sorting

### Theoretical Lower Bound

Any comparison-based sorting algorithm must be **Ω(n log n)** in the worst case.

**Why?** There are n! possible orderings. Each comparison eliminates half the possibilities. To narrow down to 1: log₂(n!) ≈ n log n comparisons needed.

---

## Simple Sorts: O(n²)

### Bubble Sort

**Idea**: Repeatedly swap adjacent elements if they're in wrong order. Large elements "bubble up" to the end.

```python
def bubble_sort(arr: list) -> list:
    """
    Time:  O(n²) average/worst, O(n) best (already sorted)
    Space: O(1)
    Stable: Yes
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        
        for j in range(n - 1 - i):  # Last i elements are sorted
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:  # Optimization: already sorted
            break
    
    return arr
```

**Step-by-step trace** for `[64, 34, 25, 12]`:

```
Pass 1: Compare adjacent pairs, largest bubbles to end
[64, 34, 25, 12] → [34, 64, 25, 12] → [34, 25, 64, 12] → [34, 25, 12, 64]
                      ↑swap                ↑swap              ↑swap

Pass 2: Second largest to second-to-last position
[34, 25, 12, 64] → [25, 34, 12, 64] → [25, 12, 34, 64]
                      ↑swap              ↑swap

Pass 3: Third largest to third-to-last position
[25, 12, 34, 64] → [12, 25, 34, 64]
                      ↑swap

Result: [12, 25, 34, 64]
```

---

### Selection Sort

**Idea**: Find the minimum element and put it first, then find second minimum, etc.

```python
def selection_sort(arr: list) -> list:
    """
    Time:  O(n²) always (always scans remaining elements)
    Space: O(1)
    Stable: No (swaps can change relative order)
    """
    n = len(arr)
    
    for i in range(n):
        min_idx = i
        
        # Find minimum in unsorted portion
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap minimum to its correct position
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr
```

**Step-by-step trace** for `[64, 25, 12, 22]`:

```
Pass 1: Find min in [64, 25, 12, 22] → 12 at index 2
        Swap with index 0: [12, 25, 64, 22]

Pass 2: Find min in [25, 64, 22] → 22 at index 3
        Swap with index 1: [12, 22, 64, 25]

Pass 3: Find min in [64, 25] → 25 at index 3
        Swap with index 2: [12, 22, 25, 64]

Result: [12, 22, 25, 64]
```

---

### Insertion Sort

**Idea**: Build sorted array one element at a time by inserting each element into its correct position.

```python
def insertion_sort(arr: list) -> list:
    """
    Time:  O(n²) average/worst, O(n) best (already sorted)
    Space: O(1)
    Stable: Yes
    
    Great for: Nearly sorted data, small arrays, online sorting
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Shift elements greater than key to the right
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr
```

**Step-by-step trace** for `[64, 25, 12, 22]`:

```
Initial: [64 | 25, 12, 22]   (left of | is "sorted")

Insert 25: [64 | 25, 12, 22]
           25 < 64, shift 64 right
           [25, 64 | 12, 22]

Insert 12: [25, 64 | 12, 22]
           12 < 64, shift 64
           12 < 25, shift 25
           [12, 25, 64 | 22]

Insert 22: [12, 25, 64 | 22]
           22 < 64, shift 64
           22 < 25, shift 25
           22 > 12, stop
           [12, 22, 25, 64]
```

---

## Efficient Sorts: O(n log n)

### Merge Sort

**Idea**: Divide array in half, recursively sort each half, merge the sorted halves.

```python
def merge_sort(arr: list) -> list:
    """
    Time:  O(n log n) always
    Space: O(n) for temporary arrays
    Stable: Yes
    
    Paradigm: Divide and Conquer
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left: list, right: list) -> list:
    """Merge two sorted arrays into one sorted array."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= for stability
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**Visual trace** for `[38, 27, 43, 3, 9, 82, 10]`:

```
                    [38, 27, 43, 3, 9, 82, 10]
                           DIVIDE
                    /                    \
           [38, 27, 43, 3]          [9, 82, 10]
              /        \              /       \
        [38, 27]    [43, 3]      [9, 82]     [10]
         /    \      /    \       /    \       |
       [38]  [27]  [43]  [3]    [9]  [82]    [10]
         \    /      \    /       \    /       |
                    MERGE
        [27, 38]    [3, 43]      [9, 82]     [10]
              \        /              \       /
           [3, 27, 38, 43]          [9, 10, 82]
                    \                    /
                    [3, 9, 10, 27, 38, 43, 82]
```

**Why O(n log n)?**
```
- log n levels (dividing by 2 each time)
- O(n) work at each level (merging)
- Total: O(n log n)
```

---

### Quick Sort

**Idea**: Choose a pivot, partition elements around it, recursively sort partitions.

```python
def quick_sort(arr: list, low: int = 0, high: int = None) -> list:
    """
    Time:  O(n log n) average, O(n²) worst (sorted input with bad pivot)
    Space: O(log n) average (stack), O(n) worst
    Stable: No
    In-place: Yes
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pivot_idx = partition(arr, low, high)
        
        # Recursively sort elements before and after pivot
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    
    return arr

def partition(arr: list, low: int, high: int) -> int:
    """
    Lomuto partition scheme.
    Places pivot at correct position, smaller elements before, larger after.
    """
    pivot = arr[high]  # Choose last element as pivot
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

**Step-by-step trace** for `[10, 80, 30, 90, 40, 50, 70]` with pivot=70:

```
Initial: [10, 80, 30, 90, 40, 50, 70]
                                  ↑ pivot=70

Partition process (i starts at -1):
j=0: arr[0]=10 ≤ 70? Yes → i=0, swap(arr[0], arr[0]): [10, 80, 30, 90, 40, 50, 70]
j=1: arr[1]=80 ≤ 70? No  → no swap
j=2: arr[2]=30 ≤ 70? Yes → i=1, swap(arr[1], arr[2]): [10, 30, 80, 90, 40, 50, 70]
j=3: arr[3]=90 ≤ 70? No  → no swap
j=4: arr[4]=40 ≤ 70? Yes → i=2, swap(arr[2], arr[4]): [10, 30, 40, 90, 80, 50, 70]
j=5: arr[5]=50 ≤ 70? Yes → i=3, swap(arr[3], arr[5]): [10, 30, 40, 50, 80, 90, 70]

Final: swap pivot into position i+1=4
[10, 30, 40, 50, 70, 90, 80]
                 ↑ pivot in correct position

Now recursively sort [10, 30, 40, 50] and [90, 80]
```

**Quick Sort Optimizations**:
```python
# 1. Median-of-three pivot selection
def median_of_three(arr, low, high):
    mid = (low + high) // 2
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    return mid

# 2. Switch to insertion sort for small subarrays
def quick_sort_optimized(arr, low, high):
    if high - low < 10:  # Small subarray
        insertion_sort(arr, low, high)
    else:
        # ... regular quicksort
        pass
```

---

### Heap Sort

**Idea**: Build a max-heap, repeatedly extract maximum to end of array.

```python
def heap_sort(arr: list) -> list:
    """
    Time:  O(n log n) always
    Space: O(1) in-place
    Stable: No
    """
    n = len(arr)
    
    # Build max heap (heapify from bottom up)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move max to end
        heapify(arr, i, 0)  # Restore heap property
    
    return arr

def heapify(arr: list, n: int, i: int):
    """Maintain max-heap property at index i."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

**Heap sort visualization**:
```
Array: [4, 10, 3, 5, 1]

Step 1: Build max heap
        10                  Array: [10, 5, 3, 4, 1]
       /  \
      5    3
     / \
    4   1

Step 2: Extract max (10), heapify
        5                   Array: [5, 4, 3, 1, | 10]
       / \
      4   3
     /
    1

Step 3: Extract max (5), heapify
        4                   Array: [4, 1, 3, | 5, 10]
       / \
      1   3

... continue until sorted: [1, 3, 4, 5, 10]
```

---

## Non-Comparison Sorting

### Counting Sort

**Idea**: Count occurrences of each value, then place elements based on counts.

```python
def counting_sort(arr: list) -> list:
    """
    Time:  O(n + k) where k = range of values
    Space: O(n + k)
    Stable: Yes (with proper implementation)
    
    Best for: Small integer range, many duplicates
    """
    if not arr:
        return arr
    
    # Find range
    min_val, max_val = min(arr), max(arr)
    range_size = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1
    
    # Reconstruct sorted array
    result = []
    for i in range(range_size):
        result.extend([i + min_val] * count[i])
    
    return result
```

**Example**: Sort `[4, 2, 2, 8, 3, 3, 1]`

```
Values: 1, 2, 3, 4, 8

Count array:
Index:  0  1  2  3  4  5  6  7
Value:  1  2  3  4  5  6  7  8
Count: [1, 2, 2, 1, 0, 0, 0, 1]

Output: [1, 2, 2, 3, 3, 4, 8]
```

---

### Radix Sort

**Idea**: Sort by each digit, from least significant to most significant.

```python
def radix_sort(arr: list) -> list:
    """
    Time:  O(d * (n + k)) where d = digits, k = base (10)
    Space: O(n + k)
    Stable: Yes (required for correctness)
    
    Best for: Large numbers with limited digits
    """
    if not arr:
        return arr
    
    max_val = max(arr)
    exp = 1  # Current digit position
    
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr: list, exp: int):
    """Sort array by digit at position exp."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    # Count occurrences of each digit
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1
    
    # Convert to cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array (traverse backwards for stability)
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy back
    for i in range(n):
        arr[i] = output[i]
```

**Example**: Sort `[170, 45, 75, 90, 802, 24, 2, 66]`

```
Pass 1 (ones digit):
[170, 90, 802, 2, 24, 45, 75, 66]
   0   0    2  2   4   5   5   6

Pass 2 (tens digit):
[802, 2, 24, 45, 66, 170, 75, 90]
   0  0   2   4   6    7   7   9

Pass 3 (hundreds digit):
[2, 24, 45, 66, 75, 90, 170, 802]
 0   0   0   0   0   0    1    8

Result: [2, 24, 45, 66, 75, 90, 170, 802]
```

---

### Bucket Sort

**Idea**: Distribute elements into buckets, sort each bucket, concatenate.

```python
def bucket_sort(arr: list, num_buckets: int = 10) -> list:
    """
    Time:  O(n + k) average, O(n²) worst (all in one bucket)
    Space: O(n + k)
    Stable: Depends on bucket sorting algorithm
    
    Best for: Uniformly distributed floating-point numbers
    """
    if not arr:
        return arr
    
    min_val, max_val = min(arr), max(arr)
    range_val = max_val - min_val
    
    # Create empty buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Distribute elements into buckets
    for num in arr:
        if range_val == 0:
            idx = 0
        else:
            idx = int((num - min_val) / range_val * (num_buckets - 1))
        buckets[idx].append(num)
    
    # Sort each bucket and concatenate
    result = []
    for bucket in buckets:
        bucket.sort()  # Or use insertion sort for small buckets
        result.extend(bucket)
    
    return result
```

---

## Sorting Algorithm Comparison

### Complexity Summary

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |
| Radix Sort | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | Yes |
| Bucket Sort | O(n+k) | O(n+k) | O(n²) | O(n+k) | Yes |

### When to Use Which?

```
┌─────────────────────────────────────────────────────────────┐
│                 SORTING ALGORITHM SELECTION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Nearly sorted data?                                        │
│  └── YES → Insertion Sort O(n)                              │
│                                                             │
│  Small array (n < 50)?                                      │
│  └── YES → Insertion Sort (low overhead)                    │
│                                                             │
│  Need stable sort?                                          │
│  └── YES → Merge Sort (guaranteed O(n log n))               │
│                                                             │
│  Memory constrained?                                        │
│  └── YES → Heap Sort (O(1) space) or Quick Sort (O(log n))  │
│                                                             │
│  Integer range small?                                       │
│  └── YES → Counting Sort O(n + k)                           │
│                                                             │
│  Large integers, limited digits?                            │
│  └── YES → Radix Sort O(d(n + k))                           │
│                                                             │
│  Uniform distribution?                                      │
│  └── YES → Bucket Sort O(n)                                 │
│                                                             │
│  General purpose?                                           │
│  └── Quick Sort (fastest in practice on average)            │
│                                                             │
│  Need guaranteed O(n log n)?                                │
│  └── Merge Sort or Heap Sort                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Python's Built-in Sort

### Timsort

Python uses **Timsort**, a hybrid of merge sort and insertion sort.

```python
# Sort in place
arr = [3, 1, 4, 1, 5, 9, 2, 6]
arr.sort()  # Modifies arr

# Return new sorted list
arr = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_arr = sorted(arr)  # arr unchanged

# Custom key function
words = ["banana", "Apple", "cherry"]
words.sort(key=str.lower)  # Case-insensitive

# Custom comparison
from functools import cmp_to_key

def compare(a, b):
    return (a > b) - (a < b)  # Return -1, 0, or 1

arr.sort(key=cmp_to_key(compare))

# Reverse order
arr.sort(reverse=True)
```

**Timsort properties**:
- Time: O(n log n) worst, O(n) best (for sorted/reverse sorted)
- Space: O(n)
- Stable: Yes
- Adaptive: Takes advantage of existing order

---

## Sorting Stability

### What Is Stability?

A **stable** sort maintains the relative order of elements with equal keys.

```
Original: [(A, 2), (B, 1), (C, 2), (D, 1)]

Stable sort by number:
         [(B, 1), (D, 1), (A, 2), (C, 2)]
          ↑        ↑       ↑        ↑
          B before D (original order preserved)
          A before C (original order preserved)

Unstable sort by number (possible result):
         [(D, 1), (B, 1), (C, 2), (A, 2)]
          Order of equal elements changed
```

### Why Stability Matters

- **Multi-key sorting**: Sort by last name, then by first name
- **Preserving existing order**: Add secondary sort without breaking primary

```python
# Sort students by grade, then by name (stable)
students = [
    ("Charlie", "A"),
    ("Alice", "B"),
    ("Bob", "A"),
    ("Diana", "B")
]

# First sort by name
students.sort(key=lambda x: x[0])
# [("Alice", "B"), ("Bob", "A"), ("Charlie", "A"), ("Diana", "B")]

# Then sort by grade (stable sort preserves name order within grades)
students.sort(key=lambda x: x[1])
# [("Bob", "A"), ("Charlie", "A"), ("Alice", "B"), ("Diana", "B")]
```

---

## Classic Interview Problems

### Problem 1: Sort Colors (Dutch National Flag)

```python
def sort_colors(nums: list[int]) -> None:
    """
    Sort array of 0s, 1s, and 2s in-place.
    Time: O(n), Space: O(1)
    """
    low, mid, high = 0, 0, len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```

### Problem 2: Merge Sorted Arrays

```python
def merge_sorted(nums1: list, m: int, nums2: list, n: int) -> None:
    """
    Merge nums2 into nums1 (which has space at end).
    Time: O(m + n), Space: O(1)
    """
    # Start from end to avoid overwriting
    i, j, k = m - 1, n - 1, m + n - 1
    
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

### Problem 3: Kth Largest Element (Quick Select)

```python
def find_kth_largest(nums: list, k: int) -> int:
    """
    Find kth largest using Quick Select.
    Time: O(n) average, O(n²) worst
    """
    k = len(nums) - k  # Convert to kth smallest
    
    def quick_select(left, right):
        pivot = nums[right]
        store = left
        
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store], nums[i] = nums[i], nums[store]
                store += 1
        
        nums[store], nums[right] = nums[right], nums[store]
        
        if store == k:
            return nums[store]
        elif store < k:
            return quick_select(store + 1, right)
        else:
            return quick_select(left, store - 1)
    
    return quick_select(0, len(nums) - 1)
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                   SORTING ALGORITHMS                         │
├─────────────────────────────────────────────────────────────┤
│ SIMPLE O(n²):                                               │
│ • Bubble: swap adjacent, good for nearly sorted             │
│ • Selection: find min repeatedly, always O(n²)              │
│ • Insertion: build sorted prefix, best for small/sorted     │
├─────────────────────────────────────────────────────────────┤
│ EFFICIENT O(n log n):                                       │
│ • Merge: divide-conquer, stable, O(n) space                 │
│ • Quick: partition-recurse, fastest average, O(n²) worst    │
│ • Heap: heapify-extract, O(1) space, not stable             │
├─────────────────────────────────────────────────────────────┤
│ NON-COMPARISON (can beat n log n):                          │
│ • Counting: O(n+k), small integer range                     │
│ • Radix: O(d(n+k)), sort by digits                          │
│ • Bucket: O(n) average, uniform distribution                │
├─────────────────────────────────────────────────────────────┤
│ PYTHON:                                                     │
│ • list.sort() - in-place, Timsort                           │
│ • sorted() - returns new list                               │
│ • key= for custom comparison                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Sort Colors | Dutch National Flag | 75 |
| 2 | Merge Sorted Array | Merge from end | 88 |
| 3 | Kth Largest Element | Quick Select | 215 |
| 4 | Sort List | Merge sort linked list | 148 |
| 5 | Largest Number | Custom comparator | 179 |
| 6 | Meeting Rooms II | Sort + heap | 253 |
| 7 | Top K Frequent Elements | Bucket sort | 347 |
| 8 | Sort Array by Parity | Two pointers | 905 |

---

## Summary

1. **Know the tradeoffs**: Time, space, stability, best/worst cases.

2. **Default choice**: Quick sort for speed, merge sort for stability.

3. **Small arrays**: Insertion sort has less overhead.

4. **Special data**: Consider non-comparison sorts (counting, radix, bucket).

5. **Interview tip**: Know how to implement merge sort and quick sort from scratch.

6. **Python**: Use `sorted()` and `list.sort()` with `key` parameter.
