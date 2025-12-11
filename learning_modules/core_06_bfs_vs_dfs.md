# CORE_06: Breadth-First Search vs Depth-First Search

## Overview

### What Are BFS and DFS?

**BFS** (Breadth-First Search) and **DFS** (Depth-First Search) are two fundamental graph/tree traversal strategies that explore nodes in different orders.

```
┌─────────────────────────────────────────────────────────────┐
│                    THE CORE DIFFERENCE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BFS: "Explore all neighbors first, then their neighbors"   │
│       → Level by level, like ripples in water               │
│       → Uses a QUEUE (FIFO)                                 │
│                                                             │
│  DFS: "Go as deep as possible, then backtrack"              │
│       → One path at a time, like exploring a maze           │
│       → Uses a STACK (LIFO) or RECURSION                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Visual Comparison

```
        Tree:
              1
            / | \
           2  3  4
          /|     |\
         5 6     7 8

BFS Order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
           (level by level)

DFS Order: 1 → 2 → 5 → 6 → 3 → 4 → 7 → 8
           (go deep, then backtrack)
```

---

## Breadth-First Search (BFS)

### Core Concept

BFS explores nodes in **layers**—all nodes at distance 1, then distance 2, then distance 3, etc.

```
         START
           ↓
    ┌──────1──────┐     Layer 0
    │      │      │
    2      3      4     Layer 1
   / \           / \
  5   6         7   8   Layer 2

BFS visits: 1, 2, 3, 4, 5, 6, 7, 8
```

### BFS Implementation

```python
from collections import deque

def bfs_graph(graph: dict, start) -> list:
    """
    BFS traversal of a graph.
    
    Time:  O(V + E) - visit each vertex and edge once
    Space: O(V) - queue and visited set
    """
    visited = {start}
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()  # FIFO: first in, first out
        result.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def bfs_tree(root) -> list:
    """
    BFS traversal of a binary tree (level-order).
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
```

### BFS Step-by-Step Trace

```
Graph: {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

Step | Queue (front→back) | Visited | Action
-----|-------------------|---------|--------
  1  | [A]               | {A}     | Start
  2  | [B, C]            | {A,B,C} | Dequeue A, add neighbors
  3  | [C, D, E]         | {A,B,C,D,E} | Dequeue B, add D,E
  4  | [D, E, F]         | {A,B,C,D,E,F} | Dequeue C, add F
  5  | [E, F]            | {A,B,C,D,E,F} | Dequeue D, no new neighbors
  6  | [F]               | {A,B,C,D,E,F} | Dequeue E, F already visited
  7  | []                | {A,B,C,D,E,F} | Dequeue F, done

Result: [A, B, C, D, E, F]
```

### BFS Level-by-Level Processing

```python
def bfs_levels(root) -> list[list]:
    """
    Return nodes grouped by level.
    
    This pattern is crucial for many tree problems!
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # KEY: snapshot current level size
        level = []
        
        for _ in range(level_size):  # Process exactly this level
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

# Example:
#       1
#      / \
#     2   3
#    / \   \
#   4   5   6
#
# Output: [[1], [2, 3], [4, 5, 6]]
```

### When to Use BFS

| Use Case | Why BFS? |
|----------|----------|
| Shortest path (unweighted) | BFS guarantees shortest by levels |
| Level-order traversal | Natural for BFS |
| Finding nearest X | First found = closest |
| Minimum steps/moves | Each level = one step |
| Social network distance | Degrees of separation |

---

## Depth-First Search (DFS)

### Core Concept

DFS explores **one path completely** before trying another path.

```
         START
           ↓
           1
          /|\
         2 3 4
        /|   |\
       5 6   7 8

DFS path: 1 → 2 → 5 → (backtrack) → 6 → (backtrack) → 3 → ...
```

### DFS Implementation (Recursive)

```python
def dfs_recursive(graph: dict, start, visited=None) -> list:
    """
    DFS traversal using recursion.
    
    Time:  O(V + E)
    Space: O(V) - recursion stack
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result

def dfs_tree(root) -> list:
    """
    DFS traversal of binary tree (preorder).
    """
    if not root:
        return []
    
    return [root.val] + dfs_tree(root.left) + dfs_tree(root.right)
```

### DFS Implementation (Iterative)

```python
def dfs_iterative(graph: dict, start) -> list:
    """
    DFS traversal using explicit stack.
    
    Time:  O(V + E)
    Space: O(V) - explicit stack
    """
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()  # LIFO: last in, first out
        
        if node in visited:
            continue
        
        visited.add(node)
        result.append(node)
        
        # Add neighbors (reverse for consistent order with recursive)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return result
```

### DFS Step-by-Step Trace

```
Graph: {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

Step | Stack (top→bottom) | Visited | Action
-----|-------------------|---------|--------
  1  | [A]               | {}      | Start
  2  | [C, B]            | {A}     | Pop A, push C, B
  3  | [C, E, D]         | {A,B}   | Pop B, push E, D
  4  | [C, E]            | {A,B,D} | Pop D, no unvisited neighbors
  5  | [C, F]            | {A,B,D,E} | Pop E, push F
  6  | [C]               | {A,B,D,E,F} | Pop F, C already in stack
  7  | []                | {A,B,C,D,E,F} | Pop C, done

Result: [A, B, D, E, F, C]
```

### DFS Variants for Trees

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# PREORDER: Root → Left → Right
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# INORDER: Left → Root → Right
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# POSTORDER: Left → Right → Root
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

```
Example Tree:
       1
      / \
     2   3
    / \
   4   5

Preorder:  [1, 2, 4, 5, 3]  (Root first)
Inorder:   [4, 2, 5, 1, 3]  (Root middle)
Postorder: [4, 5, 2, 3, 1]  (Root last)
```

### When to Use DFS

| Use Case | Why DFS? |
|----------|----------|
| Path existence | Find ANY path quickly |
| Cycle detection | Detect back edges |
| Topological sort | Process dependencies |
| Maze solving | Explore one path fully |
| Tree operations | Natural recursive structure |
| Backtracking | DFS with state restoration |
| Connected components | Mark all reachable nodes |

---

## Head-to-Head Comparison

### Traversal Order

```
              1
            / | \
           2  3  4
          /|     |\
         5 6     7 8

BFS: 1, 2, 3, 4, 5, 6, 7, 8    (Wide first)
DFS: 1, 2, 5, 6, 3, 4, 7, 8    (Deep first)
```

### Memory Usage

```
BFS Memory (Queue):
At level k, queue holds all nodes at level k
Worst case: O(b^d) where b=branching factor, d=depth

    Wide tree → Large queue
         1
     /   |   \
    2    3    4       Queue at level 1: [2, 3, 4]
   /|\ /|\ /|\
  ... ... ...         Queue at level 2: very large!

DFS Memory (Stack):
Stack holds path from root to current node
Worst case: O(d) where d=depth

    Deep tree → Long stack
    1
    |
    2           Stack at node 5: [1, 2, 3, 4, 5]
    |
    3
    |
    4
    |
    5
```

### Comparison Table

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data structure | Queue (FIFO) | Stack (LIFO) / Recursion |
| Traversal order | Level by level | Path by path |
| Memory | O(branching^depth) | O(depth) |
| Shortest path | ✓ Guaranteed (unweighted) | ✗ Not guaranteed |
| Completeness | ✓ Finds if exists (finite) | ✓ Finds if exists |
| Implementation | Iterative (usually) | Recursive (usually) |
| Good for wide trees | ✗ High memory | ✓ Lower memory |
| Good for deep trees | ✓ Lower memory | ✗ May overflow stack |

---

## Practical Decision Framework

### When to Choose BFS

```python
# 1. SHORTEST PATH in unweighted graph/grid
def shortest_path_bfs(graph, start, end):
    """BFS guarantees shortest path in unweighted graph."""
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist  # First time reaching end = shortest
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1

# 2. MINIMUM STEPS/MOVES problems
def min_knight_moves(start, end):
    """Find minimum knight moves on chess board."""
    moves = [(2,1), (2,-1), (-2,1), (-2,-1), 
             (1,2), (1,-2), (-1,2), (-1,-2)]
    
    queue = deque([(start[0], start[1], 0)])
    visited = {(start[0], start[1])}
    
    while queue:
        x, y, steps = queue.popleft()
        if (x, y) == end:
            return steps
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
    
    return -1

# 3. LEVEL-ORDER processing
def level_averages(root):
    """Average value at each level."""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        
        result.append(level_sum / level_size)
    
    return result
```

### When to Choose DFS

```python
# 1. PATH EXISTENCE (any path)
def has_path_dfs(graph, start, end, visited=None):
    """Check if any path exists."""
    if visited is None:
        visited = set()
    
    if start == end:
        return True
    
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            if has_path_dfs(graph, neighbor, end, visited):
                return True
    
    return False

# 2. CYCLE DETECTION
def has_cycle_dfs(graph):
    """Detect cycle in directed graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node):
        color[node] = GRAY  # Processing
        
        for neighbor in graph.get(node, []):
            if color[neighbor] == GRAY:
                return True  # Back edge = cycle
            if color[neighbor] == WHITE:
                if dfs(neighbor):
                    return True
        
        color[node] = BLACK  # Done
        return False
    
    return any(color[node] == WHITE and dfs(node) for node in graph)

# 3. TOPOLOGICAL SORT
def topological_sort(graph):
    """Return nodes in dependency order."""
    visited = set()
    result = []
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        
        result.append(node)  # Add AFTER visiting children
    
    for node in graph:
        dfs(node)
    
    return result[::-1]  # Reverse for correct order

# 4. TREE OPERATIONS (max depth, path sum, etc.)
def max_depth(root):
    """Maximum depth of binary tree."""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def has_path_sum(root, target_sum):
    """Check if root-to-leaf path equals target sum."""
    if not root:
        return False
    
    if not root.left and not root.right:  # Leaf
        return root.val == target_sum
    
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))

# 5. CONNECTED COMPONENTS
def count_components(graph, n):
    """Count connected components."""
    visited = set()
    count = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
    
    for i in range(n):
        if i not in visited:
            dfs(i)
            count += 1
    
    return count
```

---

## Common Interview Patterns

### Pattern 1: Multi-Source BFS

Start BFS from multiple sources simultaneously.

```python
def walls_and_gates(rooms: list[list[int]]) -> None:
    """
    Fill each empty room with distance to nearest gate.
    Gates = 0, Walls = -1, Empty = INF
    """
    if not rooms:
        return
    
    rows, cols = len(rooms), len(rooms[0])
    INF = 2147483647
    
    # Start from ALL gates (multi-source BFS)
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))
    
    # BFS from all gates simultaneously
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == INF:
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))
```

### Pattern 2: BFS with State

Track additional state beyond just position.

```python
def shortest_path_with_obstacles(grid, k):
    """
    Shortest path with ability to remove k obstacles.
    State: (row, col, obstacles_removed)
    """
    rows, cols = len(grid), len(grid[0])
    
    # State: (row, col, remaining_eliminations)
    queue = deque([(0, 0, k, 0)])  # row, col, remaining_k, steps
    visited = {(0, 0, k)}
    
    while queue:
        r, c, remaining, steps = queue.popleft()
        
        if r == rows - 1 and c == cols - 1:
            return steps
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                new_remaining = remaining - grid[nr][nc]
                
                if new_remaining >= 0 and (nr, nc, new_remaining) not in visited:
                    visited.add((nr, nc, new_remaining))
                    queue.append((nr, nc, new_remaining, steps + 1))
    
    return -1
```

### Pattern 3: Bidirectional BFS

Search from both start and end, meeting in middle.

```python
def bidirectional_bfs(graph, start, end):
    """
    BFS from both ends - can reduce O(b^d) to O(b^(d/2)).
    """
    if start == end:
        return 0
    
    # Two frontiers
    front_start = {start}
    front_end = {end}
    visited = {start, end}
    steps = 0
    
    while front_start and front_end:
        # Always expand smaller frontier (optimization)
        if len(front_start) > len(front_end):
            front_start, front_end = front_end, front_start
        
        next_frontier = set()
        
        for node in front_start:
            for neighbor in graph.get(node, []):
                if neighbor in front_end:
                    return steps + 1  # Frontiers meet!
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        
        front_start = next_frontier
        steps += 1
    
    return -1  # No path
```

### Pattern 4: DFS with Backtracking

DFS with state modification and restoration.

```python
def all_paths(graph, start, end):
    """Find all paths from start to end."""
    all_paths_result = []
    
    def dfs(node, path):
        if node == end:
            all_paths_result.append(path.copy())
            return
        
        for neighbor in graph.get(node, []):
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()  # Backtrack
    
    dfs(start, [start])
    return all_paths_result
```

---

## Grid Traversal

### BFS on Grid

```python
def bfs_grid(grid, start, end):
    """BFS on 2D grid, find shortest path."""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    queue = deque([(start[0], start[1], 0)])
    visited = {(start[0], start[1])}
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) == end:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] != '#' and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    
    return -1
```

### DFS on Grid

```python
def dfs_grid(grid, r, c, visited):
    """DFS on 2D grid (e.g., for flood fill, counting islands)."""
    rows, cols = len(grid), len(grid[0])
    
    if (r < 0 or r >= rows or c < 0 or c >= cols or 
        (r, c) in visited or grid[r][c] == '0'):
        return
    
    visited.add((r, c))
    
    # Visit all 4 neighbors
    dfs_grid(grid, r + 1, c, visited)
    dfs_grid(grid, r - 1, c, visited)
    dfs_grid(grid, r, c + 1, visited)
    dfs_grid(grid, r, c - 1, visited)

def count_islands(grid):
    """Classic DFS application: count islands."""
    if not grid:
        return 0
    
    visited = set()
    count = 0
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs_grid(grid, r, c, visited)
                count += 1
    
    return count
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                      BFS vs DFS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BFS (Breadth-First)           DFS (Depth-First)            │
│  ─────────────────             ─────────────────            │
│  Uses: QUEUE                   Uses: STACK / RECURSION      │
│  Order: Level by level         Order: Path by path          │
│  Memory: O(branching^depth)    Memory: O(depth)             │
│                                                             │
│  CHOOSE BFS FOR:               CHOOSE DFS FOR:              │
│  • Shortest path (unweighted)  • Path existence             │
│  • Minimum steps/moves         • Cycle detection            │
│  • Level-order traversal       • Topological sort           │
│  • Finding nearest             • Connected components       │
│  • Wide, shallow trees         • Tree recursion             │
│                                • Backtracking               │
│                                • Deep trees (low memory)    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ TEMPLATES:                                                  │
│                                                             │
│ BFS:                           DFS (Recursive):             │
│ queue = deque([start])         def dfs(node):               │
│ visited = {start}                  if done: return          │
│ while queue:                       visited.add(node)        │
│     node = queue.popleft()         for neighbor in adj:     │
│     for neighbor in adj:               if not visited:      │
│         if not visited:                    dfs(neighbor)    │
│             visited.add(...)                                │
│             queue.append(...)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### BFS Problems
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Binary Tree Level Order | Level processing | 102 |
| 2 | Rotting Oranges | Multi-source BFS | 994 |
| 3 | Word Ladder | Shortest transformation | 127 |
| 4 | Shortest Path in Binary Matrix | Grid BFS | 1091 |
| 5 | Open the Lock | State-space BFS | 752 |

### DFS Problems
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Number of Islands | Grid DFS | 200 |
| 2 | Max Depth of Binary Tree | Tree DFS | 104 |
| 3 | Course Schedule | Cycle detection | 207 |
| 4 | Clone Graph | Graph traversal | 133 |
| 5 | Path Sum | Root-to-leaf DFS | 112 |

### Both Work (Choose Based on Requirements)
| # | Problem | Key Concept | LeetCode # |
|---|---------|-------------|------------|
| 1 | Flood Fill | Grid traversal | 733 |
| 2 | Symmetric Tree | Tree comparison | 101 |
| 3 | All Paths From Source | Graph paths | 797 |

---

## Summary

1. **BFS = Queue, Level-by-level**: Best for shortest path and minimum steps.

2. **DFS = Stack/Recursion, Path-by-path**: Best for existence checks and tree operations.

3. **Both are O(V + E)**: Same time complexity, but different memory characteristics.

4. **Memory tradeoff**: BFS uses more memory for wide graphs, DFS for deep graphs.

5. **Shortest path**: BFS guarantees shortest in unweighted graphs; DFS does not.

6. **Default choice**: 
   - Need shortest? → BFS
   - Tree recursion? → DFS
   - Backtracking? → DFS
   - Level processing? → BFS
