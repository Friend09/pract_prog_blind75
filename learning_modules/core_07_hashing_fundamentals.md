# CORE_07: Hashing Fundamentals

## Overview

### What Is Hashing?

**Hashing** is a technique that maps data of arbitrary size to fixed-size values (hash codes). This enables O(1) average-time operations for insertion, deletion, and lookup.

```
┌─────────────────────────────────────────────────────────────┐
│                   HASHING IN ONE SENTENCE                   │
├─────────────────────────────────────────────────────────────┤
│  "Convert any key into an array index for instant access."  │
│                                                             │
│  Key: "apple" ──→ hash("apple") ──→ 42 ──→ array[42]       │
│                                                             │
│  Without hashing: O(n) search                               │
│  With hashing:    O(1) average lookup                       │
└─────────────────────────────────────────────────────────────┘
```

### Why Hashing Matters

| Without Hash Table | With Hash Table |
|-------------------|-----------------|
| Array search: O(n) | Lookup: O(1) avg |
| BST search: O(log n) | Insert: O(1) avg |
| Find duplicate: O(n²) | Find duplicate: O(n) |
| Two Sum: O(n²) | Two Sum: O(n) |

---

## Hash Functions

### What Is a Hash Function?

A **hash function** takes an input (key) and produces a fixed-size integer (hash code).

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    Key      │ ───→ │   Hash      │ ───→ │   Index     │
│  "hello"    │      │  Function   │      │     7       │
└─────────────┘      └─────────────┘      └─────────────┘

hash("hello") = 99162322
index = 99162322 % table_size = 7
```

### Properties of a Good Hash Function

| Property | Description | Why It Matters |
|----------|-------------|----------------|
| **Deterministic** | Same input → same output | Consistent lookups |
| **Uniform** | Distributes keys evenly | Minimizes collisions |
| **Efficient** | Fast to compute | O(1) operations |
| **Avalanche** | Small input change → big hash change | Better distribution |

### Common Hash Functions

```python
# Simple hash for strings (educational)
def simple_hash(key: str, table_size: int) -> int:
    """
    Sum of character ASCII values, then mod table size.
    Problem: "abc" and "cba" produce same hash!
    """
    return sum(ord(c) for c in key) % table_size

# Polynomial rolling hash (better distribution)
def polynomial_hash(key: str, table_size: int, base: int = 31) -> int:
    """
    Treats string as base-31 number.
    "abc" = a*31² + b*31¹ + c*31⁰
    Much better distribution than simple sum.
    """
    hash_val = 0
    for char in key:
        hash_val = (hash_val * base + ord(char)) % table_size
    return hash_val

# Python's built-in hash
def python_hash_example():
    """
    Python's hash() is optimized and varies by type.
    Note: hash() can return negative values!
    """
    print(hash("hello"))      # Integer hash code
    print(hash(42))           # Numbers hash to themselves (mostly)
    print(hash((1, 2, 3)))    # Tuples are hashable
    # hash([1, 2, 3])         # Error! Lists are mutable, not hashable
```

### Hash Function for Integers

```python
def hash_integer(key: int, table_size: int) -> int:
    """
    Simple modulo hash for integers.
    Table size should be prime for better distribution.
    """
    return key % table_size

# Why prime table size?
# Consider table_size = 10, keys: 10, 20, 30, 40, 50
# All hash to 0! Bad distribution.
# 
# With table_size = 11 (prime):
# 10→10, 20→9, 30→8, 40→7, 50→6
# Much better distribution!
```

---

## Hash Tables

### Structure of a Hash Table

A **hash table** (or hash map) stores key-value pairs using an array, with positions determined by hashing.

```
Hash Table Structure:

    Index   Value
    ┌───┬─────────────┐
  0 │   │ ("dog", 3)  │
    ├───┼─────────────┤
  1 │   │   None      │
    ├───┼─────────────┤
  2 │   │ ("cat", 5)  │
    ├───┼─────────────┤
  3 │   │ ("bird", 2) │
    ├───┼─────────────┤
  4 │   │   None      │
    └───┴─────────────┘

hash("dog") % 5 = 0 → stored at index 0
hash("cat") % 5 = 2 → stored at index 2
hash("bird") % 5 = 3 → stored at index 3
```

### Basic Hash Table Operations

```python
class SimpleHashTable:
    """
    Basic hash table (without collision handling).
    For educational purposes only!
    """
    def __init__(self, size: int = 10):
        self.size = size
        self.table = [None] * size
    
    def _hash(self, key: str) -> int:
        """Compute index from key."""
        return hash(key) % self.size
    
    def put(self, key: str, value) -> None:
        """Insert or update key-value pair. O(1)"""
        index = self._hash(key)
        self.table[index] = (key, value)
    
    def get(self, key: str):
        """Retrieve value by key. O(1)"""
        index = self._hash(key)
        if self.table[index] and self.table[index][0] == key:
            return self.table[index][1]
        return None
    
    def remove(self, key: str) -> None:
        """Remove key-value pair. O(1)"""
        index = self._hash(key)
        if self.table[index] and self.table[index][0] == key:
            self.table[index] = None
```

---

## Collision Handling

### What Is a Collision?

A **collision** occurs when two different keys hash to the same index.

```
Collision Example:
hash("cat") % 5 = 2
hash("dog") % 5 = 2   ← COLLISION!

Both keys want index 2. Now what?
```

### Method 1: Chaining (Separate Chaining)

Each array slot holds a linked list (or other collection) of all entries that hash to that index.

```
Chaining Structure:

Index   Chain (Linked List)
┌───┬──────────────────────────────────┐
  0 │ → [("apple", 1)] → None          │
├───┼──────────────────────────────────┤
  1 │ → None                           │
├───┼──────────────────────────────────┤
  2 │ → [("cat", 5)] → [("dog", 3)] → None  ← Both hash to 2
├───┼──────────────────────────────────┤
  3 │ → [("bird", 2)] → None           │
└───┴──────────────────────────────────┘
```

```python
class HashTableChaining:
    """
    Hash table with chaining for collision resolution.
    
    Time:  O(1) average, O(n) worst (all keys collide)
    Space: O(n)
    """
    def __init__(self, size: int = 10):
        self.size = size
        self.table = [[] for _ in range(size)]  # List of lists
        self.count = 0
    
    def _hash(self, key) -> int:
        return hash(key) % self.size
    
    def put(self, key, value) -> None:
        """Insert or update. O(1) average."""
        index = self._hash(key)
        
        # Check if key exists, update if so
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        
        # Key doesn't exist, append
        self.table[index].append((key, value))
        self.count += 1
        
        # Resize if load factor too high
        if self.count / self.size > 0.75:
            self._resize()
    
    def get(self, key):
        """Retrieve value. O(1) average."""
        index = self._hash(key)
        
        for k, v in self.table[index]:
            if k == key:
                return v
        return None
    
    def remove(self, key) -> bool:
        """Remove key. O(1) average."""
        index = self._hash(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                self.count -= 1
                return True
        return False
    
    def _resize(self) -> None:
        """Double table size and rehash all entries."""
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]
        self.count = 0
        
        for chain in old_table:
            for key, value in chain:
                self.put(key, value)
```

### Method 2: Open Addressing

All entries stored directly in the array. On collision, probe for next empty slot.

```
Open Addressing (Linear Probing):

Insert "dog" (hash=2), "cat" (hash=2), "rat" (hash=2):

Step 1: Insert "dog"          Step 2: Insert "cat"      Step 3: Insert "rat"
┌───┬─────────────┐           ┌───┬─────────────┐       ┌───┬─────────────┐
  0 │   None      │             0 │   None      │         0 │   None      │
  1 │   None      │             1 │   None      │         1 │   None      │
  2 │   "dog"     │ ←           2 │   "dog"     │         2 │   "dog"     │
  3 │   None      │             3 │   "cat"     │ ← probe  3 │   "cat"     │
  4 │   None      │             4 │   None      │         4 │   "rat"     │ ← probe
└───┴─────────────┘           └───┴─────────────┘       └───┴─────────────┘
```

```python
class HashTableOpenAddressing:
    """
    Hash table with linear probing.
    
    Time:  O(1) average, O(n) worst
    Space: O(n)
    """
    def __init__(self, size: int = 10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0
        self.DELETED = object()  # Sentinel for deleted slots
    
    def _hash(self, key) -> int:
        return hash(key) % self.size
    
    def _probe(self, key) -> int:
        """Find slot for key using linear probing."""
        index = self._hash(key)
        first_deleted = None
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return index  # Found existing key
            
            if self.keys[index] is self.DELETED and first_deleted is None:
                first_deleted = index  # Remember first deleted slot
            
            index = (index + 1) % self.size  # Linear probe
        
        # Return first deleted slot if found, else empty slot
        return first_deleted if first_deleted is not None else index
    
    def put(self, key, value) -> None:
        """Insert or update. O(1) average."""
        if self.count / self.size > 0.5:  # Keep load factor low
            self._resize()
        
        index = self._probe(key)
        
        if self.keys[index] is None or self.keys[index] is self.DELETED:
            self.count += 1
        
        self.keys[index] = key
        self.values[index] = value
    
    def get(self, key):
        """Retrieve value. O(1) average."""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
        
        return None
    
    def remove(self, key) -> bool:
        """Remove key (mark as deleted). O(1) average."""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.keys[index] = self.DELETED
                self.values[index] = None
                self.count -= 1
                return True
            index = (index + 1) % self.size
        
        return False
    
    def _resize(self) -> None:
        """Double size and rehash."""
        old_keys, old_values = self.keys, self.values
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        
        for i, key in enumerate(old_keys):
            if key is not None and key is not self.DELETED:
                self.put(key, old_values[i])
```

### Probing Strategies

```
LINEAR PROBING:
index = (hash + i) % size
Probe sequence: 0, 1, 2, 3, 4, ...
Problem: Clustering (consecutive filled slots)

QUADRATIC PROBING:
index = (hash + i²) % size
Probe sequence: 0, 1, 4, 9, 16, ...
Problem: Secondary clustering

DOUBLE HASHING:
index = (hash1 + i * hash2) % size
Uses second hash function for step size
Best distribution, but more complex
```

### Chaining vs Open Addressing

| Aspect | Chaining | Open Addressing |
|--------|----------|-----------------|
| Implementation | Simpler | More complex |
| Memory | Extra for pointers | No extra pointers |
| Load factor | Can exceed 1.0 | Must stay < 1.0 |
| Cache performance | Poor (pointer chasing) | Better (contiguous) |
| Deletion | Simple | Requires tombstones |
| Worst case | Long chains | Clustering |

---

## Load Factor and Resizing

### What Is Load Factor?

**Load factor** = (number of entries) / (table size)

```
Load Factor = n / m

n = 7 entries
m = 10 slots
Load factor = 0.7 (70% full)

Higher load factor → More collisions → Slower operations
Lower load factor  → Wasted space   → Faster operations
```

### When to Resize

| Method | Resize Threshold | Target Load After |
|--------|-----------------|-------------------|
| Chaining | 0.75 | 0.375 |
| Open Addressing | 0.5 | 0.25 |

```python
def check_and_resize(self):
    """Resize when load factor exceeds threshold."""
    load_factor = self.count / self.size
    
    if load_factor > 0.75:  # Threshold
        self._resize(self.size * 2)  # Double size
```

### Rehashing Process

```
Before resize (size=4, load=0.75):
┌───┬────────────┐
  0 │ ("a", 1)   │
  1 │ ("b", 2)   │
  2 │ None       │
  3 │ ("c", 3)   │
└───┴────────────┘

After resize (size=8):
All entries rehashed to new positions!

┌───┬────────────┐
  0 │ None       │
  1 │ ("a", 1)   │  ← hash("a") % 8 = 1
  2 │ ("b", 2)   │  ← hash("b") % 8 = 2
  3 │ None       │
  4 │ None       │
  5 │ ("c", 3)   │  ← hash("c") % 8 = 5
  6 │ None       │
  7 │ None       │
└───┴────────────┘

Rehashing is O(n) but happens infrequently
→ Amortized O(1) per operation
```

---

## Python's Hash-Based Collections

### dict (Hash Map)

```python
# Dictionary: key-value storage with O(1) operations
d = {}
d["apple"] = 5        # O(1) insert
print(d["apple"])     # O(1) lookup → 5
del d["apple"]        # O(1) delete
print(d.get("banana", 0))  # O(1) with default → 0

# Common patterns
d = {"a": 1, "b": 2}
"a" in d              # O(1) membership test → True
d.keys()              # View of keys
d.values()            # View of values
d.items()             # View of (key, value) pairs

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

### set (Hash Set)

```python
# Set: unique elements with O(1) operations
s = set()
s.add(5)              # O(1) insert
s.add(5)              # No duplicate added
print(5 in s)         # O(1) membership → True
s.remove(5)           # O(1) delete (raises KeyError if missing)
s.discard(5)          # O(1) delete (no error if missing)

# Set from list (removes duplicates)
nums = [1, 2, 2, 3, 3, 3]
unique = set(nums)    # {1, 2, 3}

# Set operations
a = {1, 2, 3}
b = {2, 3, 4}
a | b                 # Union: {1, 2, 3, 4}
a & b                 # Intersection: {2, 3}
a - b                 # Difference: {1}
a ^ b                 # Symmetric difference: {1, 4}
```

### collections.Counter

```python
from collections import Counter

# Count occurrences
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})

counter["apple"]        # 3
counter["orange"]       # 0 (missing keys return 0)
counter.most_common(2)  # [('apple', 3), ('banana', 2)]

# Useful operations
counter.update(["apple", "date"])  # Add more counts
counter.subtract(["apple"])        # Subtract counts

# Count characters in string
char_count = Counter("mississippi")
# Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
```

### collections.defaultdict

```python
from collections import defaultdict

# Default values for missing keys
d = defaultdict(list)
d["fruits"].append("apple")   # No KeyError, creates empty list first
d["fruits"].append("banana")
# {'fruits': ['apple', 'banana']}

d = defaultdict(int)
d["count"] += 1               # No KeyError, defaults to 0
# {'count': 1}

d = defaultdict(set)
d["seen"].add(5)              # Creates empty set first
# {'seen': {5}}

# Graph adjacency list
graph = defaultdict(list)
edges = [(0, 1), (0, 2), (1, 2)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)
# {0: [1, 2], 1: [0, 2], 2: [0, 1]}
```

---

## Classic Hash Table Problems

### Problem 1: Two Sum

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Find indices of two numbers that sum to target.
    
    Time:  O(n) - single pass with hash lookup
    Space: O(n) - hash map storage
    
    Without hash: O(n²) brute force
    """
    seen = {}  # value → index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:  # O(1) lookup
            return [seen[complement], i]
        
        seen[num] = i  # O(1) insert
    
    return []

# Example:
# nums = [2, 7, 11, 15], target = 9
# i=0: complement=7, seen={}, add 2→0
# i=1: complement=2, seen={2:0}, found! return [0, 1]
```

### Problem 2: Contains Duplicate

```python
def contains_duplicate(nums: list[int]) -> bool:
    """
    Check if array contains any duplicate.
    
    Time:  O(n)
    Space: O(n)
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False

# One-liner using set size comparison
def contains_duplicate_oneliner(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))
```

### Problem 3: Group Anagrams

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Group strings that are anagrams of each other.
    
    Time:  O(n * k log k) where k = max string length
    Space: O(n * k)
    
    Key insight: Anagrams have the same sorted characters
    """
    groups = defaultdict(list)
    
    for s in strs:
        key = tuple(sorted(s))  # Anagrams share this key
        groups[key].append(s)
    
    return list(groups.values())

# Alternative: Use character count as key (O(n * k))
def group_anagrams_count(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    
    for s in strs:
        # Count of each letter as key
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        key = tuple(count)
        groups[key].append(s)
    
    return list(groups.values())

# Example:
# ["eat", "tea", "tan", "ate", "nat", "bat"]
# → [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

### Problem 4: Longest Consecutive Sequence

```python
def longest_consecutive(nums: list[int]) -> int:
    """
    Find length of longest consecutive sequence.
    
    Time:  O(n)
    Space: O(n)
    
    Key insight: Only start counting from sequence starts
    """
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Only start if num is the START of a sequence
        if num - 1 not in num_set:
            current = num
            length = 1
            
            while current + 1 in num_set:
                current += 1
                length += 1
            
            longest = max(longest, length)
    
    return longest

# Example:
# nums = [100, 4, 200, 1, 3, 2]
# 1 is start → 1, 2, 3, 4 → length 4
# 100 is start → 100 → length 1
# 200 is start → 200 → length 1
# Answer: 4
```

### Problem 5: LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used Cache.
    
    All operations O(1):
    - get: Move to end (most recent)
    - put: Add/update and move to end
    - Evict: Remove from front (least recent)
    
    Uses OrderedDict which maintains insertion order
    and supports O(1) move_to_end.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        if len(self.cache) > self.capacity:
            # Remove oldest (first item)
            self.cache.popitem(last=False)
```

### Problem 6: Subarray Sum Equals K

```python
def subarray_sum(nums: list[int], k: int) -> int:
    """
    Count subarrays that sum to k.
    
    Time:  O(n)
    Space: O(n)
    
    Key insight: Use prefix sums with hash map
    If prefix[j] - prefix[i] = k, subarray [i+1, j] sums to k
    So we look for prefix[j] - k in our seen prefix sums
    """
    count = 0
    prefix_sum = 0
    prefix_counts = {0: 1}  # Empty prefix has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # How many times have we seen (prefix_sum - k)?
        if prefix_sum - k in prefix_counts:
            count += prefix_counts[prefix_sum - k]
        
        # Record this prefix sum
        prefix_counts[prefix_sum] = prefix_counts.get(prefix_sum, 0) + 1
    
    return count

# Example:
# nums = [1, 1, 1], k = 2
# prefix sums: 0, 1, 2, 3
# At 2: look for 0 → found once
# At 3: look for 1 → found once
# Answer: 2 (subarrays [1,1] at positions 0-1 and 1-2)
```

---

## Design Considerations

### Choosing Hash Table Size

```python
# Prefer prime numbers for table size
# They provide better distribution with modulo hashing

GOOD_SIZES = [11, 23, 47, 97, 193, 389, 769, 1543, 3079, 6151, 
              12289, 24593, 49157, 98317, 196613, 393241, 786433]

def next_prime_size(n):
    """Find next prime >= n."""
    for size in GOOD_SIZES:
        if size >= n:
            return size
    return n * 2 + 1  # Fallback
```

### Hashable Objects in Python

```python
# Hashable: immutable objects with __hash__ method
hashable_types = [
    42,              # int
    3.14,            # float
    "hello",         # str
    (1, 2, 3),       # tuple (with hashable elements)
    frozenset([1, 2])  # frozenset
]

# NOT hashable: mutable objects
# list, dict, set → Cannot be dictionary keys

# Custom hashable class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))  # Hash tuple of attributes
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# Now Point can be used as dict key or set element
points = {Point(0, 0): "origin", Point(1, 1): "diagonal"}
```

### Hash Table vs Other Data Structures

| Operation | Hash Table | Array | BST | Sorted Array |
|-----------|------------|-------|-----|--------------|
| Search | O(1) avg | O(n) | O(log n) | O(log n) |
| Insert | O(1) avg | O(1)* | O(log n) | O(n) |
| Delete | O(1) avg | O(n) | O(log n) | O(n) |
| Min/Max | O(n) | O(n) | O(log n) | O(1) |
| Ordered iteration | O(n log n) | O(n) | O(n) | O(n) |
| Space | O(n) | O(n) | O(n) | O(n) |

*At end; O(n) at arbitrary position

---

## Common Mistakes

### Mistake 1: Mutable Keys

```python
# WRONG: Using mutable object as key
d = {}
key = [1, 2, 3]
d[key] = "value"  # TypeError: unhashable type: 'list'

# FIX: Convert to immutable type
d = {}
key = tuple([1, 2, 3])
d[key] = "value"  # Works!
```

### Mistake 2: Not Handling Missing Keys

```python
# WRONG: KeyError if key missing
d = {"a": 1}
print(d["b"])  # KeyError!

# FIX 1: Use .get() with default
print(d.get("b", 0))  # 0

# FIX 2: Use defaultdict
from collections import defaultdict
d = defaultdict(int)
print(d["b"])  # 0, and adds "b": 0 to dict

# FIX 3: Check membership first
if "b" in d:
    print(d["b"])
```

### Mistake 3: Modifying Dict While Iterating

```python
# WRONG: Modifying size during iteration
d = {"a": 1, "b": 2, "c": 3}
for key in d:
    if d[key] < 2:
        del d[key]  # RuntimeError!

# FIX: Iterate over copy of keys
d = {"a": 1, "b": 2, "c": 3}
for key in list(d.keys()):
    if d[key] < 2:
        del d[key]  # Works!
```

### Mistake 4: Hash Collisions in Custom Objects

```python
# WRONG: Same hash for all objects
class BadPoint:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __hash__(self):
        return 42  # All objects hash to 42!
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# All operations become O(n) due to collisions!

# FIX: Proper hash using all relevant attributes
class GoodPoint:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __hash__(self):
        return hash((self.x, self.y))  # Unique hash for each point
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                  HASHING FUNDAMENTALS                        │
├─────────────────────────────────────────────────────────────┤
│ HASH TABLE OPERATIONS:                                      │
│ • Insert: O(1) average, O(n) worst                          │
│ • Lookup: O(1) average, O(n) worst                          │
│ • Delete: O(1) average, O(n) worst                          │
├─────────────────────────────────────────────────────────────┤
│ COLLISION HANDLING:                                         │
│ • Chaining: Linked lists at each slot                       │
│ • Open Addressing: Probe for next empty slot                │
│   - Linear: (hash + i) % size                               │
│   - Quadratic: (hash + i²) % size                           │
│   - Double: (hash1 + i*hash2) % size                        │
├─────────────────────────────────────────────────────────────┤
│ LOAD FACTOR:                                                │
│ • α = n / m (entries / slots)                               │
│ • Chaining: resize at α > 0.75                              │
│ • Open addressing: resize at α > 0.5                        │
├─────────────────────────────────────────────────────────────┤
│ PYTHON COLLECTIONS:                                         │
│ • dict: key-value mapping                                   │
│ • set: unique elements                                      │
│ • Counter: count occurrences                                │
│ • defaultdict: auto-create missing keys                     │
├─────────────────────────────────────────────────────────────┤
│ COMMON PATTERNS:                                            │
│ • Two Sum: complement lookup                                │
│ • Group by: sorted key or count tuple                       │
│ • Prefix sum: prefix_sum - k lookup                         │
│ • Duplicates: set membership                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Two Sum | Hash lookup | 1 |
| 2 | Contains Duplicate | Set membership | 217 |
| 3 | Valid Anagram | Character count | 242 |
| 4 | Group Anagrams | Hash grouping | 49 |
| 5 | Longest Consecutive | Set + sequence | 128 |
| 6 | Top K Frequent Elements | Counter | 347 |
| 7 | Subarray Sum Equals K | Prefix sum + hash | 560 |
| 8 | LRU Cache | OrderedDict | 146 |
| 9 | Happy Number | Cycle detection | 202 |
| 10 | First Unique Character | Count map | 387 |

---

## Summary

1. **O(1) average**: Hash tables provide constant-time operations on average.

2. **Collision handling**: Chaining (simpler) vs. open addressing (cache-friendly).

3. **Load factor**: Keep low (< 0.75) to maintain performance. Resize when exceeded.

4. **Python tools**: Use `dict`, `set`, `Counter`, `defaultdict` as appropriate.

5. **Common patterns**: Two Sum, grouping, prefix sums, deduplication.

6. **Hash requirements**: Keys must be hashable (immutable) and implement `__hash__` and `__eq__`.
