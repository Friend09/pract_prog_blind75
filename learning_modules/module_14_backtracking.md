# Module 14: Backtracking

## Pattern Overview

### The Core Problem This Pattern Solves

Backtracking solves problems where you need to **explore all possible configurations** to find valid solutions. These are problems where you're building something step-by-step, and at each step, you have multiple choices—but some choices lead to dead ends.

**The fundamental challenge**: How do you systematically explore a massive solution space without:
1. Missing any valid solutions
2. Wasting time on paths that can't possibly lead to solutions
3. Getting lost in the exponential number of possibilities

**Imagine trying to solve a Sudoku puzzle without backtracking**... you'd have to try every possible combination of numbers (9^81 possibilities ≈ 10^77). With backtracking, you place a number, immediately check if it violates any rules, and if it does, you "backtrack" and try a different number. This **pruning** eliminates vast swaths of the search space.

### When to Recognize This Pattern

**Problem Signals** (keywords and characteristics that hint at backtracking):

| Signal                               | Example Problem Statement                                 |
| ------------------------------------ | --------------------------------------------------------- |
| "Find ALL possible..."               | "Find all permutations of the array"                      |
| "Generate ALL valid..."              | "Generate all valid parentheses combinations"             |
| "List EVERY way to..."               | "List every way to partition the string into palindromes" |
| "Count the number of ways..."        | "Count ways to reach the target sum"                      |
| "Can you construct/place/arrange..." | "Can you place N queens on an N×N board?"                 |
| "Path finding with constraints"      | "Find all paths from source to destination"               |
| Combinatorial problems               | Subsets, permutations, combinations                       |
| Constraint satisfaction              | Sudoku, N-Queens, crossword puzzles                       |

**Input Characteristics**:
- Discrete choices at each decision point
- Constraints that must be satisfied
- Solution is built incrementally
- Multiple valid solutions may exist (or we need to find just one)
- Solution space is too large for brute force enumeration

### Real-World Analogy

**The Maze Explorer**

Imagine you're exploring a maze to find all exits:

```
┌───────────────────────────┐
│ START                     │
│   ↓                       │
│   ├──→ Dead End           │
│   │                       │
│   ├──→ ├──→ Dead End      │
│   │    │                  │
│   │    └──→ EXIT 1 ✓      │
│   │                       │
│   └──→ ├──→ EXIT 2 ✓      │
│        │                  │
│        └──→ Dead End      │
└───────────────────────────┘
```

1. **CHOOSE**: At each fork, pick a direction
2. **EXPLORE**: Walk down that path
3. **DEAD END?**: If you hit a wall, turn back to the last fork
4. **BACKTRACK**: Try the next untried direction
5. **FOUND EXIT?**: Record it, then backtrack to find more exits

**What the analogy captures well**:
- The systematic exploration of all paths
- The "undo" action when hitting dead ends
- The recursive nature (each fork creates sub-mazes)

**Where the analogy breaks down**:
- Real backtracking often involves **pruning** (knowing a path is bad before fully exploring it)
- We often build a "solution" as we go, not just navigate

---

## Key Concepts at a Glance

| Term                 | Definition                                                     | Why It Matters                                        |
| -------------------- | -------------------------------------------------------------- | ----------------------------------------------------- |
| **State Space Tree** | A tree representing all possible states/choices in the problem | Visualizes the entire search space we're exploring    |
| **Decision Point**   | A position where we must choose from multiple options          | Each recursive call handles one decision              |
| **Constraint**       | A rule that valid solutions must satisfy                       | Allows us to prune invalid branches early             |
| **Pruning**          | Skipping entire subtrees that can't lead to valid solutions    | The key optimization that makes backtracking feasible |
| **Partial Solution** | The solution built so far (passed through recursion)           | What we're incrementally constructing                 |
| **Base Case**        | Condition indicating we've found a complete solution           | Tells us when to stop recursing and record the answer |
| **Backtrack**        | Undoing the last choice to try alternatives                    | The "undo" operation that lets us explore all paths   |

---

## The Pattern Mechanics

### Core Idea in One Sentence

> **Build a solution incrementally, abandoning a path ("backtracking") as soon as you determine it cannot lead to a valid solution.**

### Visual Representation: State Space Tree

For generating all subsets of `[1, 2, 3]`:

```
                        []
                    ┌───┴───┐
               include 1?  exclude 1?
                  /            \
                [1]            []
              ┌──┴──┐       ┌──┴──┐
         incl 2  excl 2  incl 2  excl 2
           /      \        /       \
        [1,2]    [1]     [2]       []
        /   \    /  \    /  \     /  \
      [1,2,3][1,2][1,3][1] [2,3][2] [3] []
         ↑     ↑    ↑   ↑    ↑   ↑   ↑   ↑
         └─────┴────┴───┴────┴───┴───┴───┘
                  All 8 subsets (2³)
```

### The Backtracking Template

```python
def backtrack(candidate, state, results):
    """
    Generic backtracking template.

    Args:
        candidate: Current partial solution being built
        state: Current position/index in the problem
        results: Collection of all valid solutions found
    """
    # BASE CASE: Is this a complete, valid solution?
    if is_complete(candidate):
        results.append(candidate.copy())  # IMPORTANT: copy!
        return

    # GENERATE CHOICES: What options do we have at this state?
    for choice in get_choices(state):

        # CONSTRAINT CHECK (PRUNING): Is this choice valid?
        if not is_valid(candidate, choice):
            continue  # Skip invalid choices (PRUNE)

        # MAKE CHOICE: Add choice to our partial solution
        candidate.append(choice)  # or modify state

        # RECURSE: Explore further with this choice
        backtrack(candidate, next_state(state), results)

        # BACKTRACK: Undo the choice (restore state)
        candidate.pop()  # or undo state modification
```

### The Three Key Operations

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKTRACKING CYCLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐        │
│    │  CHOOSE  │ ───→ │  EXPLORE │ ───→ │   UNDO   │        │
│    │          │      │          │      │          │        │
│    │ Add to   │      │ Recurse  │      │ Remove   │        │
│    │ solution │      │ deeper   │      │ from     │        │
│    │          │      │          │      │ solution │        │
│    └──────────┘      └──────────┘      └──────────┘        │
│         ↑                                    │              │
│         └────────────────────────────────────┘              │
│                    (try next choice)                        │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works (The Intuition)

**Exhaustive but Smart Search**:

1. **Completeness**: By systematically trying every choice at every decision point, we guarantee we don't miss any valid solution.

2. **Efficiency through Pruning**: By checking constraints early (before fully exploring a path), we eliminate entire subtrees of invalid solutions.

3. **State Recovery**: By undoing each choice after exploring it, we restore the state to try alternative choices—this is the "backtrack" that gives the pattern its name.

**The Mathematical Guarantee**:
- If we explore all branches (depth-first traversal of state space tree)
- And we correctly identify valid solutions at leaves
- Then we find ALL valid solutions

---

## Concrete Example with Full Trace

### Problem: Generate All Permutations

**Problem Statement**: Given an array of distinct integers, return all possible permutations.

**Input**: `nums = [1, 2, 3]`
**Output**: `[[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]`

### Brute Force Approach (What We're Improving)

```python
# Naive: Generate all arrangements and filter valid ones
# This doesn't really improve on backtracking for permutations,
# but for constrained problems, brute force would check ALL n!
# permutations even when many violate constraints.

from itertools import permutations

def brute_force(nums):
    return list(permutations(nums))  # Generates all n!
```

**Why backtracking is better for constrained problems**: Brute force generates everything first, then filters. Backtracking prunes during generation.

### Optimal Solution Using Backtracking

```python
def permute(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations using backtracking.

    Time Complexity: O(n! * n) - n! permutations, O(n) to copy each
    Space Complexity: O(n) - recursion depth + current path
    """
    results = []

    def backtrack(current_perm: list[int], remaining: set[int]):
        # BASE CASE: All numbers used → complete permutation
        if len(current_perm) == len(nums):
            results.append(current_perm.copy())
            return

        # Try each remaining number
        for num in list(remaining):  # list() to avoid modification during iteration
            # CHOOSE: Add num to current permutation
            current_perm.append(num)
            remaining.remove(num)

            # EXPLORE: Recurse with updated state
            backtrack(current_perm, remaining)

            # BACKTRACK: Undo the choice
            current_perm.pop()
            remaining.add(num)

    backtrack([], set(nums))
    return results
```

### Detailed Execution Trace

**Input**: `nums = [1, 2, 3]`

```
Legend:
  current_perm = path being built
  remaining = numbers still available
  → = recursive call
  ← = backtrack (return)
  ★ = solution found
```

| Step | Action       | current_perm | remaining | Notes                      |
| ---- | ------------ | ------------ | --------- | -------------------------- |
| 1    | Start        | `[]`         | `{1,2,3}` | Initial call               |
| 2    | Choose 1     | `[1]`        | `{2,3}`   | Try first number           |
| 3    | → Choose 2   | `[1,2]`      | `{3}`     | Recurse, try 2             |
| 4    | → → Choose 3 | `[1,2,3]`    | `{}`      | Recurse, try 3             |
| 5    | ★ FOUND      | `[1,2,3]`    | `{}`      | Base case! Record solution |
| 6    | ← Backtrack  | `[1,2]`      | `{3}`     | Remove 3, restore {3}      |
| 7    | ← Backtrack  | `[1]`        | `{2,3}`   | Remove 2, restore {2}      |
| 8    | Choose 3     | `[1,3]`      | `{2}`     | Try 3 instead              |
| 9    | → Choose 2   | `[1,3,2]`    | `{}`      | Only 2 remains             |
| 10   | ★ FOUND      | `[1,3,2]`    | `{}`      | Record solution            |
| 11   | ← Backtrack  | `[1,3]`      | `{2}`     | Remove 2                   |
| 12   | ← Backtrack  | `[1]`        | `{2,3}`   | Remove 3                   |
| 13   | ← Backtrack  | `[]`         | `{1,2,3}` | Remove 1, try next         |
| 14   | Choose 2     | `[2]`        | `{1,3}`   | Start with 2               |
| ...  | ...          | ...          | ...       | Continue similarly...      |

### Visual State Space Tree

```
                              backtrack([], {1,2,3})
                    ┌──────────────┼──────────────┐
                 pick 1         pick 2         pick 3
                    │              │              │
            ([1], {2,3})    ([2], {1,3})    ([3], {1,2})
              ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
           pick 2   pick 3  pick 1  pick 3  pick 1  pick 2
              │        │       │       │       │       │
        ([1,2],{3}) ([1,3],{2}) ...   ...     ...     ...
              │        │
           pick 3   pick 2
              │        │
         [1,2,3]★  [1,3,2]★    [2,1,3]★ [2,3,1]★ [3,1,2]★ [3,2,1]★
```

---

## Complexity Analysis

### Time Complexity

**For Permutations (no pruning)**:
- **Nodes in tree**: 1 + n + n(n-1) + n(n-1)(n-2) + ... + n! = O(n × n!)
- **Work per node**: O(1) for choose/unchoose, O(n) to copy solution
- **Total**: O(n! × n)

**General Backtracking**:
```
Time = O(nodes_explored × work_per_node)

Without pruning: nodes = size of entire state space
With pruning:    nodes = size of valid state space (often MUCH smaller)
```

**Derivation for Subsets**:
```
Tree has 2^n leaves (each element included or not)
Internal nodes: 2^n - 1
Total nodes: 2^(n+1) - 1 = O(2^n)
Copying each subset: O(n) average
Total: O(n × 2^n)
```

### Space Complexity

**Recursion Stack**: O(n) where n = maximum depth of recursion

**Auxiliary Space**:
- Current path/candidate: O(n)
- Results storage: O(solutions × solution_size)

**What uses space**:
| Component          | Space    | Reason                          |
| ------------------ | -------- | ------------------------------- |
| Recursion stack    | O(depth) | One frame per recursive call    |
| Current candidate  | O(n)     | Building solution incrementally |
| Results            | O(k × n) | k solutions of avg size n       |
| Used/remaining set | O(n)     | Tracking available choices      |

---

## Pattern Variations

### Variation 1: Permutations with Duplicates

**When to use**: Input contains duplicate elements, but output shouldn't have duplicate permutations.

**Key difference**: Sort input, skip duplicates at same level.

```python
def permute_unique(nums: list[int]) -> list[list[int]]:
    """
    Permutations of array with duplicates.
    Key insight: Skip duplicate numbers at the same decision level.
    """
    results = []
    nums.sort()  # CRUCIAL: sorting groups duplicates together
    used = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            results.append(current.copy())
            return

        for i in range(len(nums)):
            # Skip if already used
            if used[i]:
                continue

            # Skip duplicates at same level:
            # If nums[i] == nums[i-1] and nums[i-1] wasn't used,
            # then we're trying the same value again at this level
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return results

# Example: [1, 1, 2]
# Without dedup: [[1,1,2], [1,2,1], [1,1,2], [1,2,1], [2,1,1], [2,1,1]]
# With dedup:    [[1,1,2], [1,2,1], [2,1,1]]  ✓
```

### Variation 2: Combinations (Choose k from n)

**When to use**: Order doesn't matter, fixed size selection.

**Key difference**: Only consider elements after current index (no looking back).

```python
def combine(n: int, k: int) -> list[list[int]]:
    """
    Generate all combinations of k numbers from [1, n].

    Key insight: To avoid duplicates like [1,2] and [2,1],
    only pick numbers greater than the last picked number.
    """
    results = []

    def backtrack(start: int, current: list[int]):
        # Base case: combination complete
        if len(current) == k:
            results.append(current.copy())
            return

        # Optimization: if remaining numbers < needed, prune
        remaining = n - start + 1
        needed = k - len(current)
        if remaining < needed:
            return  # Can't possibly complete, prune!

        # Only consider numbers >= start (prevents duplicates)
        for num in range(start, n + 1):
            current.append(num)
            backtrack(num + 1, current)  # start from num+1, not start+1
            current.pop()

    backtrack(1, [])
    return results
```

### Variation 3: Subsets (Power Set)

**When to use**: Generate all possible subsets.

**Key difference**: Every node is a valid solution (not just leaves).

```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets. Every partial solution is valid!
    """
    results = []

    def backtrack(start: int, current: list[int]):
        # Every current state is a valid subset
        results.append(current.copy())

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return results
```

### Variation 4: Constraint Satisfaction (N-Queens, Sudoku)

**When to use**: Finding arrangements that satisfy multiple constraints.

**Key difference**: Heavy pruning based on constraint violations.

```python
def solve_n_queens(n: int) -> list[list[str]]:
    """
    Place n queens on n×n board so no two attack each other.

    Key insight: Use sets to track attacked columns and diagonals
    for O(1) constraint checking.
    """
    results = []

    # Track which columns and diagonals are under attack
    cols = set()         # Column index
    diag1 = set()        # row - col (identifies one diagonal)
    diag2 = set()        # row + col (identifies other diagonal)

    board = [['.'] * n for _ in range(n)]

    def backtrack(row: int):
        if row == n:
            # All queens placed successfully
            results.append([''.join(r) for r in board])
            return

        for col in range(n):
            # Check all constraints in O(1)
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue  # This position is under attack, PRUNE

            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            # Recurse to next row
            backtrack(row + 1)

            # Remove queen (backtrack)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return results
```

---

## Classic Problems Using This Pattern

### Problem 1: Subsets (Easy → Medium)

**Problem**: Given an integer array `nums` of unique elements, return all possible subsets.

**Key Insight**: Include/exclude decision for each element; every path is valid.

```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Time: O(n × 2^n) - 2^n subsets, O(n) to copy each
    Space: O(n) - recursion depth
    """
    results = []

    def backtrack(index: int, current: list[int]):
        # Every state is a valid subset
        results.append(current.copy())

        # Try including each remaining element
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return results
```

**Edge Cases**:
- Empty input → `[[]]`
- Single element → `[[], [elem]]`

---

### Problem 2: Combination Sum (Medium)

**Problem**: Find all unique combinations where candidate numbers sum to target. Same number can be used unlimited times.

**Key Insight**: Allow reusing same index (unlimited use), but move forward to avoid permutation duplicates.

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Time: O(n^(target/min)) - worst case branching
    Space: O(target/min) - max recursion depth
    """
    results = []
    candidates.sort()  # Enables pruning

    def backtrack(start: int, current: list[int], remaining: int):
        if remaining == 0:
            results.append(current.copy())
            return

        for i in range(start, len(candidates)):
            num = candidates[i]

            # Pruning: if current number > remaining, all future numbers will too
            if num > remaining:
                break

            current.append(num)
            # Pass i (not i+1) to allow reuse of same number
            backtrack(i, current, remaining - num)
            current.pop()

    backtrack(0, [], target)
    return results

# Example: candidates=[2,3,6,7], target=7
# Output: [[2,2,3], [7]]
```

---

### Problem 3: Letter Combinations of Phone Number (Medium)

**Problem**: Given digits 2-9, return all possible letter combinations.

**Key Insight**: Each digit maps to letters; build combinations character by character.

```python
def letter_combinations(digits: str) -> list[str]:
    """
    Time: O(4^n × n) - up to 4 letters per digit, n digits
    Space: O(n) - recursion depth
    """
    if not digits:
        return []

    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    results = []

    def backtrack(index: int, current: list[str]):
        if index == len(digits):
            results.append(''.join(current))
            return

        for letter in phone_map[digits[index]]:
            current.append(letter)
            backtrack(index + 1, current)
            current.pop()

    backtrack(0, [])
    return results
```

---

### Problem 4: Palindrome Partitioning (Medium)

**Problem**: Partition string so every substring is a palindrome.

**Key Insight**: At each position, try all possible palindrome prefixes, then recurse on remainder.

```python
def partition(s: str) -> list[list[str]]:
    """
    Time: O(n × 2^n) - 2^(n-1) ways to partition, O(n) palindrome check
    Space: O(n) - recursion depth
    """
    results = []

    def is_palindrome(sub: str) -> bool:
        return sub == sub[::-1]

    def backtrack(start: int, current: list[str]):
        if start == len(s):
            results.append(current.copy())
            return

        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]

            # Only continue if this part is a palindrome (PRUNE)
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    return results

# Example: "aab" → [["a","a","b"], ["aa","b"]]
```

---

### Problem 5: Word Search (Medium)

**Problem**: Find if word exists in 2D grid, moving horizontally/vertically adjacent.

**Key Insight**: DFS from each cell with backtracking; mark visited cells.

```python
def exist(board: list[list[str]], word: str) -> bool:
    """
    Time: O(m × n × 4^L) - L = word length, 4 directions
    Space: O(L) - recursion depth
    """
    rows, cols = len(board), len(board[0])

    def backtrack(row: int, col: int, index: int) -> bool:
        # Base case: found entire word
        if index == len(word):
            return True

        # Boundary and character check
        if (row < 0 or row >= rows or
            col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False

        # Mark as visited (modify in place as "backtrack state")
        temp = board[row][col]
        board[row][col] = '#'  # Mark visited

        # Explore all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                 backtrack(row - 1, col, index + 1) or
                 backtrack(row, col + 1, index + 1) or
                 backtrack(row, col - 1, index + 1))

        # Backtrack: restore cell
        board[row][col] = temp

        return found

    # Try starting from each cell
    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True

    return False
```

---

## Edge Cases and Gotchas

### Edge Case 1: Empty Input

**Scenario**: Input array/string is empty
**Input Example**: `nums = []`
**Expected Output**: `[[]]` for subsets, `[]` for permutations of fixed length

```python
def handle_empty(nums):
    if not nums:
        return [[]]  # Subsets: empty set is a valid subset
        # OR return []  # Permutations: no permutations possible
```

### Edge Case 2: Single Element

**Scenario**: Only one element in input
**Input Example**: `nums = [1]`
**Expected Output**: `[[1]]` for permutations, `[[], [1]]` for subsets

### Edge Case 3: Duplicates in Input

**Scenario**: Input has duplicate values, output shouldn't have duplicate combinations
**Input Example**: `nums = [1, 1, 2]`
**How to Handle**: Sort + skip same-value siblings

```python
# In the loop:
if i > start and nums[i] == nums[i-1]:
    continue  # Skip duplicate at same level
```

### Edge Case 4: Large Search Space (Performance)

**Scenario**: Input is large, brute force will TLE
**How to Handle**: Aggressive pruning + early termination

```python
# Prune impossible branches early
if remaining_capacity < 0:
    return  # Can't possibly succeed

# Prune when not enough elements remain
if n - index < k - len(current):
    return  # Not enough elements to complete combination
```

### Common Mistakes

| Mistake                              | Why It Happens                             | How to Avoid                                 |
| ------------------------------------ | ------------------------------------------ | -------------------------------------------- |
| Forgetting to copy result            | Appending reference to mutable list        | Always use `result.copy()` or `list(result)` |
| Not backtracking                     | Forgetting to undo choice after recursion  | Always pair "make choice" with "undo choice" |
| Modifying collection while iterating | Changing the set/list being looped over    | Create a copy: `for x in list(collection):`  |
| Off-by-one in indices                | Confusion about inclusive/exclusive bounds | Draw out small examples, trace carefully     |
| Missing base case                    | Recursion never terminates                 | Identify: "What makes a complete solution?"  |
| Duplicate results                    | Not handling duplicate input values        | Sort input + skip same-value siblings        |

---

## Common Misconceptions

### ❌ MISCONCEPTION: "Backtracking is just brute force with extra steps"

✅ **REALITY**: Backtracking with pruning can be **exponentially faster** than brute force. For N-Queens on an 8×8 board:
- Brute force: Check all 64^8 = 2.8 × 10^14 configurations
- Backtracking: Explores only ~2,057 configurations (one queen per row + constraint pruning)

**WHY THE CONFUSION**: Without pruning, backtracking IS brute force. The power comes from constraint checking.

**INTERVIEW TIP**: Always discuss pruning strategies and how they reduce the search space.

---

### ❌ MISCONCEPTION: "The order of trying choices doesn't matter"

✅ **REALITY**: Order can affect:
1. **Whether you find a solution faster** (for "find any" problems)
2. **The order of results** (may need to be sorted/consistent)
3. **Pruning effectiveness** (trying smaller numbers first enables better pruning)

**INTERVIEW TIP**: Mention that sorting input often helps with pruning and duplicate handling.

---

### ❌ MISCONCEPTION: "Backtracking always has exponential time complexity"

✅ **REALITY**:
- **With good pruning**: Can be much faster in practice
- **Problem-dependent**: Some problems have polynomial solutions with clever pruning
- **Average vs. Worst case**: Average case often much better than worst case

**INTERVIEW TIP**: Discuss both theoretical worst case AND expected behavior with pruning.

---

## Interview Tips for This Pattern

### How to Communicate Your Approach

```
1. IDENTIFY THE PATTERN:
   "This is a backtracking problem because we need to explore all
   possible [combinations/permutations/configurations] that satisfy
   [constraint]."

2. STATE SPACE:
   "The state space tree has [description]. Each node represents
   [partial solution state]."

3. CHOICES AT EACH STEP:
   "At each step, we choose from [available options]. Our choices
   are constrained by [constraints]."

4. PRUNING STRATEGY:
   "We can prune branches early by checking [constraint] before
   recursing. This eliminates [what gets eliminated]."

5. COMPLEXITY:
   "The time complexity is O([complexity]) because [reasoning].
   With pruning, the average case is better."
```

### Common Follow-Up Questions

| Follow-Up Question                       | What They're Testing      | How to Respond                                       |
| ---------------------------------------- | ------------------------- | ---------------------------------------------------- |
| "How would you optimize this?"           | Pruning knowledge         | Discuss constraint propagation, ordering heuristics  |
| "What if there are duplicates?"          | Edge case handling        | Sort + skip siblings with same value                 |
| "Can you do it iteratively?"             | Stack-based understanding | Yes, using explicit stack (but recursive is cleaner) |
| "How would you parallelize?"             | Systems thinking          | Each top-level choice can be explored independently  |
| "What's the actual runtime in practice?" | Beyond Big-O              | Discuss pruning effectiveness, average case          |

### Red Flags to Avoid

- **Jumping to code without explaining state space**: Show you understand the problem structure first
- **Forgetting to backtrack**: The most common bug—always undo your choice
- **Not discussing pruning**: Shows you might not optimize well
- **Ignoring duplicates**: Often leads to wrong answers in interviews
- **Claiming O(1) space when recursion uses O(n)**: Recursion stack counts!

---

## Comparison with Related Patterns

| Aspect            | Backtracking                           | DFS                       | BFS                        | Dynamic Programming        |
| ----------------- | -------------------------------------- | ------------------------- | -------------------------- | -------------------------- |
| **Best for**      | All solutions, constraint satisfaction | Single path, connectivity | Shortest path (unweighted) | Optimal solution, counting |
| **Explores**      | All valid paths                        | All reachable nodes       | Level by level             | Subproblems once           |
| **Key operation** | Make/undo choice                       | Visit/mark                | Enqueue neighbors          | Memoize results            |
| **Time**          | O(b^d) worst                           | O(V+E)                    | O(V+E)                     | O(subproblems × work)      |
| **Pruning**       | Essential                              | Optional                  | Rare                       | Via problem structure      |
| **Solution type** | All solutions                          | Any path                  | Shortest path              | One optimal                |

### Decision Tree: When to Use What?

```
Do you need ALL solutions?
├── Yes → Is there a DP substructure?
│   ├── Yes → DP might count solutions faster
│   └── No → BACKTRACKING ✓
└── No → Do you need the OPTIMAL solution?
    ├── Yes → Is there optimal substructure?
    │   ├── Yes → DYNAMIC PROGRAMMING
    │   └── No → Backtracking with early termination
    └── No → Just need ANY solution?
        ├── Shortest path? → BFS
        └── Any path? → DFS
```

---

## Real-World Applications

### Application 1: Compilers (Register Allocation)

**Context**: Assigning variables to a limited number of CPU registers
**How Backtracking Applies**: Try assigning variables to registers; backtrack when conflicts arise
**Why It Matters**: Efficient register allocation = faster compiled code

### Application 2: Scheduling Systems

**Context**: Assigning tasks to time slots/resources with constraints
**How Backtracking Applies**: Try assignments, backtrack when constraints violated
**Examples**: Meeting room scheduling, exam timetabling, airline crew scheduling

### Application 3: Game AI

**Context**: Finding solutions to puzzles (Sudoku solvers, crossword generators)
**How Backtracking Applies**: Try each possible value, backtrack on invalid states
**Why It Matters**: Powers puzzle games and automated puzzle generation

### Application 4: Configuration Systems

**Context**: Finding valid configurations in complex systems
**How Backtracking Applies**: Incrementally build configuration, backtrack on conflicts
**Examples**: Network routing, circuit design, dependency resolution

---

## Practice Problems (Ordered by Difficulty)

### Warm-Up (Easy)
| #   | Problem                 | Key Concept                  | LeetCode # |
| --- | ----------------------- | ---------------------------- | ---------- |
| 1   | Subsets                 | Basic include/exclude        | 78         |
| 2   | Binary Watch            | Combinations with constraint | 401        |
| 3   | Letter Case Permutation | Character-level choices      | 784        |

### Core Practice (Medium)
| #   | Problem                             | Key Concept                    | LeetCode # |
| --- | ----------------------------------- | ------------------------------ | ---------- |
| 1   | Permutations                        | Classic permutation            | 46         |
| 2   | Permutations II                     | Handling duplicates            | 47         |
| 3   | Combinations                        | k-element selection            | 77         |
| 4   | Combination Sum                     | Unlimited reuse                | 39         |
| 5   | Combination Sum II                  | Each element once + duplicates | 40         |
| 6   | Subsets II                          | Subsets with duplicates        | 90         |
| 7   | Palindrome Partitioning             | String partitioning            | 131        |
| 8   | Letter Combinations of Phone Number | Mapping + combinations         | 17         |
| 9   | Generate Parentheses                | Valid sequence generation      | 22         |
| 10  | Word Search                         | Grid-based backtracking        | 79         |

### Challenge (Hard)
| #   | Problem                  | Key Concept                  | LeetCode # |
| --- | ------------------------ | ---------------------------- | ---------- |
| 1   | N-Queens                 | Constraint satisfaction      | 51         |
| 2   | N-Queens II              | Counting solutions           | 52         |
| 3   | Sudoku Solver            | Heavy constraint propagation | 37         |
| 4   | Word Search II           | Trie + backtracking          | 212        |
| 5   | Expression Add Operators | Operators + evaluation       | 282        |

### Recommended Practice Order
1. **Subsets** (78) - Learn basic structure
2. **Permutations** (46) - Classic template
3. **Combinations** (77) - Forward-only traversal
4. **Combination Sum** (39) - Pruning with target
5. **Permutations II** (47) - Duplicate handling
6. **Generate Parentheses** (22) - Constraint tracking
7. **Word Search** (79) - Grid application
8. **N-Queens** (51) - Complex constraints
9. **Sudoku Solver** (37) - Ultimate constraint satisfaction

---

## Code Templates

### Template 1: Basic Backtracking (Subsets/Combinations)

```python
def backtrack_template_subsets(nums: list[int]) -> list[list[int]]:
    """
    Template for subset/combination generation.

    Customize:
    - Base case condition
    - What constitutes a valid result
    - Whether to record at every node or only leaves
    """
    results = []

    def backtrack(start: int, current: list[int]):
        # Record current state (for subsets, every state is valid)
        results.append(current.copy())

        # Try each element from start onwards
        for i in range(start, len(nums)):
            # Optional: skip duplicates
            # if i > start and nums[i] == nums[i-1]:
            #     continue

            current.append(nums[i])
            backtrack(i + 1, current)  # i+1 = no reuse; i = allow reuse
            current.pop()

    # nums.sort()  # Uncomment if handling duplicates
    backtrack(0, [])
    return results
```

### Template 2: Permutations with Tracking

```python
def backtrack_template_permutations(nums: list[int]) -> list[list[int]]:
    """
    Template for permutation generation.

    Customize:
    - Used tracking mechanism (set, boolean array, or in-place swap)
    - Duplicate handling strategy
    """
    results = []
    used = [False] * len(nums)

    def backtrack(current: list[int]):
        if len(current) == len(nums):
            results.append(current.copy())
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            # Skip duplicates (requires sorted input)
            # if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
            #     continue

            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    # nums.sort()  # Uncomment if handling duplicates
    backtrack([])
    return results
```

### Template 3: Constraint Satisfaction (Grid-based)

```python
def backtrack_template_grid(grid: list[list[any]]) -> bool:
    """
    Template for grid-based constraint satisfaction (Sudoku, N-Queens).

    Customize:
    - What constitutes a valid placement
    - Constraint checking logic
    - How to find next empty cell
    """
    def is_valid(row: int, col: int, value: any) -> bool:
        # Check all constraints for placing value at (row, col)
        # Return True if valid, False otherwise
        pass

    def find_empty() -> tuple[int, int] | None:
        # Find next cell to fill
        # Return (row, col) or None if complete
        pass

    def backtrack() -> bool:
        pos = find_empty()
        if pos is None:
            return True  # All cells filled = solution found

        row, col = pos

        for value in get_possible_values():
            if is_valid(row, col, value):
                grid[row][col] = value

                if backtrack():
                    return True

                grid[row][col] = EMPTY  # Backtrack

        return False  # No valid value found

    return backtrack()
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                      BACKTRACKING                           │
├─────────────────────────────────────────────────────────────┤
│ WHEN TO USE:                                                │
│ • "Find ALL possible..."                                    │
│ • "Generate all valid..."                                   │
│ • Constraint satisfaction problems                          │
│ • Combinatorial: subsets, permutations, combinations        │
├─────────────────────────────────────────────────────────────┤
│ KEY IDEA: Build incrementally, backtrack on dead ends       │
├─────────────────────────────────────────────────────────────┤
│ COMPLEXITY:                                                 │
│ Time: O(b^d) worst case, often better with pruning          │
│ Space: O(d) for recursion depth                             │
├─────────────────────────────────────────────────────────────┤
│ TEMPLATE:                                                   │
│   1. Base case: Is solution complete? → Record it           │
│   2. For each choice:                                       │
│      a. Is it valid? (Prune if not)                         │
│      b. Make choice                                         │
│      c. Recurse                                             │
│      d. Undo choice (BACKTRACK)                             │
├─────────────────────────────────────────────────────────────┤
│ WATCH OUT FOR:                                              │
│ • Forgetting to copy results (use .copy())                  │
│ • Not undoing choices (the actual "backtrack")              │
│ • Duplicate results (sort + skip same-value siblings)       │
│ • Off-by-one errors in indices                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Connections to Other Patterns

### Prerequisites
- **Recursion** - Backtracking is inherently recursive
- **DFS** - Backtracking is DFS on the state space tree
- **Trees** - Understanding tree traversal helps visualize state space

### Builds Upon
- Extends basic recursion with state tracking and undoing
- Combines DFS traversal with constraint checking

### Leads To
- **Dynamic Programming** - When subproblems overlap, memoize backtracking
- **Branch and Bound** - Adds optimization (best solution so far) to pruning
- **Constraint Programming** - Formalized constraint satisfaction

### Often Combined With
- **Trie + Backtracking** for word search problems (LC 212)
- **Graph + Backtracking** for path finding with constraints
- **Memoization + Backtracking** → becomes DP when subproblems overlap

---

## Summary: Key Takeaways

1. **Core Principle**: Systematically explore all possibilities by building solutions incrementally and abandoning paths that can't lead to valid solutions.

2. **When to Use**: Problems asking for "all possible" solutions, combinations, permutations, or constraint satisfaction.

3. **Key Insight**: The power of backtracking comes from **pruning**—cutting off branches early saves exponential time.

4. **Common Mistake**: Forgetting to undo your choice after recursion (the actual "backtrack").

5. **Interview Tip**: Always discuss (1) the state space tree structure, (2) your pruning strategy, and (3) how you handle duplicates.

---

## Additional Resources

### Video Explanations
- **Back To Back SWE** - Excellent visual explanations of backtracking
- **NeetCode** - Problem-focused walkthroughs with code

### Reading
- **"Cracking the Coding Interview"** Chapter on Recursion and Backtracking
- **"Algorithm Design Manual"** by Skiena - Chapter 7 on Backtracking

### Interactive Practice
- **LeetCode** - Filter by "Backtracking" tag, sort by acceptance rate
- **Visualgo.net** - Visualize recursion trees
