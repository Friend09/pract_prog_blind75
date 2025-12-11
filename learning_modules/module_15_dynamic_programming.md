# Module 15: Dynamic Programming

## Pattern Overview

### The Core Problem This Pattern Solves

Dynamic Programming (DP) solves problems that have **overlapping subproblems** and **optimal substructure**. Without DP, you'd solve the same subproblems repeatedly, leading to exponential time complexity.

**Imagine trying to calculate Fibonacci(50) with naive recursion:**
- You'd compute Fibonacci(48) twice, Fibonacci(47) three times, Fibonacci(46) five times...
- Total recursive calls: ~2^50 ≈ 1,125,899,906,842,624 operations
- With DP: exactly 50 operations

**What types of problems become solvable with DP?**
- Optimization problems: "Find the minimum/maximum..."
- Counting problems: "How many ways to..."
- Decision problems: "Is it possible to..."
- Problems with recursive structure where subproblems repeat

**The brute force approach (why it fails):**
- Enumerate all possibilities (often exponential: 2^n or n!)
- Solve same subproblems repeatedly
- Time complexity explodes for even moderate inputs

### When to Recognize This Pattern

**Problem Signals** (keywords and characteristics):
- **"Find the minimum/maximum..."**: Minimum cost, maximum profit, longest/shortest path
- **"Count the number of ways..."**: Ways to reach target, distinct paths, combinations
- **"Is it possible to..."**: Can you partition, can you reach, can you form
- **"Find the longest/shortest..."**: Longest subsequence, shortest path
- **Choices at each step**: Take or skip, include or exclude, go left or right

**Input Characteristics**:
- Sequential data (arrays, strings) where position matters
- Problems with clear state that can be parameterized
- Decisions that affect future options
- Constraints that suggest polynomial (not exponential) solution exists

**Two Key Properties (MUST have both):**

| Property                    | Definition                                                 | Example                                                   |
| --------------------------- | ---------------------------------------------------------- | --------------------------------------------------------- |
| **Optimal Substructure**    | Optimal solution contains optimal solutions to subproblems | Shortest path A→C through B = shortest A→B + shortest B→C |
| **Overlapping Subproblems** | Same subproblems are solved multiple times                 | Fib(5) needs Fib(4) and Fib(3); Fib(4) also needs Fib(3)  |

### Real-World Analogy

**Planning a Road Trip with Multiple Stops**

Imagine driving from New York to Los Angeles, and you want the cheapest route. You must pass through Chicago and Denver.

- **Optimal Substructure**: The cheapest NY→LA route through Chicago and Denver = (cheapest NY→Chicago) + (cheapest Chicago→Denver) + (cheapest Denver→LA)
- **Overlapping Subproblems**: If you're also calculating NY→Seattle (through Chicago and Denver), you reuse NY→Chicago and NY→Denver calculations

**What the analogy captures well:**
- Breaking big problems into smaller, independent pieces
- Reusing calculations you've already done
- Building up to the final answer step by step

**Where the analogy breaks down:**
- Real DP problems often have more complex state than just "current location"
- The "subproblems" in DP are often more abstract

---

## Key Concepts at a Glance

| Term                    | Definition                                        | Why It Matters                                                 |
| ----------------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| **State**               | Variables that uniquely define a subproblem       | Determines what you memoize; wrong state = wrong answer        |
| **Transition**          | How to compute current state from previous states | The recurrence relation; the heart of DP                       |
| **Base Case**           | Smallest subproblems with known answers           | Where recursion stops; starting point for tabulation           |
| **Memoization**         | Top-down: cache results of recursive calls        | Avoids recomputation; natural recursive thinking               |
| **Tabulation**          | Bottom-up: fill table iteratively                 | Often faster; no recursion overhead; easier space optimization |
| **State Space**         | All possible combinations of state variables      | Determines time/space complexity                               |
| **Recurrence Relation** | Mathematical formula relating states              | dp[i] = f(dp[i-1], dp[i-2], ...)                               |

---

## The Pattern Mechanics

### Core Idea in One Sentence

**Store solutions to subproblems so you never solve the same subproblem twice, then combine these solutions to solve larger problems.**

### The Two Approaches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMIC PROGRAMMING                                 │
├─────────────────────────────────┬───────────────────────────────────────────┤
│         TOP-DOWN                │              BOTTOM-UP                    │
│       (Memoization)             │             (Tabulation)                  │
├─────────────────────────────────┼───────────────────────────────────────────┤
│  • Start from main problem      │  • Start from base cases                  │
│  • Recursively break down       │  • Iteratively build up                   │
│  • Cache results in hashmap     │  • Fill table systematically              │
│  • Lazy: only compute needed    │  • Eager: compute all states              │
│  • Natural recursive thinking   │  • Often more efficient                   │
│  • Risk of stack overflow       │  • Easier to optimize space               │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### Visual: Fibonacci Example

```
                    Naive Recursion (Exponential)
                    ─────────────────────────────
                              fib(5)
                            /        \
                       fib(4)        fib(3)
                      /     \        /     \
                  fib(3)  fib(2)  fib(2)  fib(1)
                  /    \
              fib(2)  fib(1)

    Notice: fib(3) computed 2 times, fib(2) computed 3 times!


                    With Memoization (Linear)
                    ─────────────────────────
                              fib(5)
                            /        \
                       fib(4)        fib(3) ← CACHED!
                      /     \
                  fib(3)  fib(2) ← CACHED!
                  /    \
              fib(2)  fib(1)

    Each subproblem solved exactly once, then retrieved from cache.


                    Tabulation (Linear, No Recursion)
                    ─────────────────────────────────

    dp[0] = 0    dp[1] = 1    (base cases)

    Index:    0     1     2     3     4     5
            ┌─────┬─────┬─────┬─────┬─────┬─────┐
    dp:     │  0  │  1  │  1  │  2  │  3  │  5  │
            └─────┴─────┴─────┴─────┴─────┴─────┘
                          ↑     ↑     ↑     ↑
                         0+1   1+1   1+2   2+3

    Fill left to right: dp[i] = dp[i-1] + dp[i-2]
```

### The 5-Step DP Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE 5-STEP DP FRAMEWORK                                  │
├──────┬──────────────────────────────────────────────────────────────────────┤
│ STEP │ ACTION                                                               │
├──────┼──────────────────────────────────────────────────────────────────────┤
│  1   │ DEFINE STATE                                                         │
│      │ • What variables uniquely identify a subproblem?                     │
│      │ • What does dp[i] or dp[i][j] represent?                             │
│      │ • Write it in plain English first!                                   │
├──────┼──────────────────────────────────────────────────────────────────────┤
│  2   │ IDENTIFY BASE CASES                                                  │
│      │ • What are the smallest subproblems?                                 │
│      │ • What values are known without computation?                         │
│      │ • Usually: dp[0], dp[1], or boundary conditions                      │
├──────┼──────────────────────────────────────────────────────────────────────┤
│  3   │ WRITE RECURRENCE RELATION                                            │
│      │ • How does dp[i] relate to smaller subproblems?                      │
│      │ • What choices/decisions lead to current state?                      │
│      │ • dp[i] = f(dp[i-1], dp[i-2], ...) or min/max of options             │
├──────┼──────────────────────────────────────────────────────────────────────┤
│  4   │ DETERMINE ITERATION ORDER                                            │
│      │ • Which direction fills the table correctly?                         │
│      │ • Ensure dependencies are computed before needed                     │
│      │ • Usually: left→right, top→bottom, or diagonal                       │
├──────┼──────────────────────────────────────────────────────────────────────┤
│  5   │ LOCATE THE ANSWER                                                    │
│      │ • Where in the table is the final answer?                            │
│      │ • Usually: dp[n], dp[n][m], or max/min of entire table               │
└──────┴──────────────────────────────────────────────────────────────────────┘
```

### Why This Works (The Intuition)

**Mathematical Foundation: Principle of Optimality**

If you have an optimal solution to a problem, then the solution to any subproblem within that solution must also be optimal.

**Why we don't miss valid solutions:**
- We systematically explore ALL ways to reach each state
- At each state, we keep only the optimal (min/max) or count all possibilities
- The recurrence relation guarantees we consider every valid path

**Why it's efficient:**
- Each state computed exactly once: O(number of states)
- Each computation takes O(transitions per state)
- Total: O(states × transitions), usually polynomial

---

## Concrete Example 1: Climbing Stairs (1D DP)

### Problem Statement

You are climbing a staircase with `n` steps. Each time you can climb 1 or 2 steps. In how many distinct ways can you reach the top?

**Input**: `n = 5`
**Output**: `8`
**Constraints**: `1 <= n <= 45`

### Applying the 5-Step Framework

**Step 1: Define State**
```
dp[i] = number of distinct ways to reach step i
```

**Step 2: Base Cases**
```
dp[0] = 1  (one way to stay at ground: do nothing)
dp[1] = 1  (one way to reach step 1: take one step)
```

**Step 3: Recurrence Relation**
```
To reach step i, you either:
  - Came from step i-1 (took 1 step)
  - Came from step i-2 (took 2 steps)

dp[i] = dp[i-1] + dp[i-2]
```

**Step 4: Iteration Order**
```
Left to right: i = 2, 3, 4, ..., n
(We need dp[i-1] and dp[i-2] before computing dp[i])
```

**Step 5: Answer Location**
```
dp[n] = number of ways to reach step n
```

### Brute Force Approach

```python
def climb_stairs_brute(n: int) -> int:
    """
    Time: O(2^n) - binary tree of recursive calls
    Space: O(n) - recursion stack depth
    """
    if n <= 1:
        return 1
    return climb_stairs_brute(n - 1) + climb_stairs_brute(n - 2)
```

**Why it's slow**: Creates a binary tree of calls, with massive redundancy.

### Top-Down (Memoization) Solution

```python
def climb_stairs_memo(n: int) -> int:
    """
    Count distinct ways to climb n stairs (1 or 2 steps at a time).

    Time Complexity: O(n) - each state computed once
    Space Complexity: O(n) - memo dict + recursion stack
    """
    memo = {}

    def dp(i: int) -> int:
        # Base cases
        if i <= 1:
            return 1

        # Check cache
        if i in memo:
            return memo[i]

        # Compute and cache
        memo[i] = dp(i - 1) + dp(i - 2)
        return memo[i]

    return dp(n)
```

### Bottom-Up (Tabulation) Solution

```python
def climb_stairs_tabulation(n: int) -> int:
    """
    Count distinct ways to climb n stairs (1 or 2 steps at a time).

    Time Complexity: O(n) - single pass through array
    Space Complexity: O(n) - dp array
    """
    if n <= 1:
        return 1

    # Initialize dp array
    dp = [0] * (n + 1)
    dp[0] = 1  # Base case: 1 way to stay at ground
    dp[1] = 1  # Base case: 1 way to reach step 1

    # Fill table left to right
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]
```

### Space-Optimized Solution

```python
def climb_stairs_optimized(n: int) -> int:
    """
    Count distinct ways to climb n stairs.

    Time Complexity: O(n)
    Space Complexity: O(1) - only track last two values
    """
    if n <= 1:
        return 1

    prev2 = 1  # dp[i-2], starts as dp[0]
    prev1 = 1  # dp[i-1], starts as dp[1]

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1
```

### Detailed Execution Trace

**Input**: `n = 5`

| Step | i   | dp[i-2] | dp[i-1] | Calculation | dp[i] | Array State            |
| ---- | --- | ------- | ------- | ----------- | ----- | ---------------------- |
| Init | -   | -       | -       | Base cases  | -     | [1, 1, 0, 0, 0, 0]     |
| 1    | 2   | dp[0]=1 | dp[1]=1 | 1 + 1       | 2     | [1, 1, **2**, 0, 0, 0] |
| 2    | 3   | dp[1]=1 | dp[2]=2 | 1 + 2       | 3     | [1, 1, 2, **3**, 0, 0] |
| 3    | 4   | dp[2]=2 | dp[3]=3 | 2 + 3       | 5     | [1, 1, 2, 3, **5**, 0] |
| 4    | 5   | dp[3]=3 | dp[4]=5 | 3 + 5       | 8     | [1, 1, 2, 3, 5, **8**] |

**Final Output**: `dp[5] = 8`

### Visual State at Each Step

```
Step 0 (Base Cases):
Stairs:  [0]──[1]──[2]──[3]──[4]──[5]
Ways:     1    1    ?    ?    ?    ?

Step 1 (i=2):
To reach step 2: come from step 0 (2-step jump) OR step 1 (1-step)
         1 way + 1 way = 2 ways
Stairs:  [0]──[1]──[2]──[3]──[4]──[5]
Ways:     1    1    2    ?    ?    ?

Step 2 (i=3):
To reach step 3: come from step 1 OR step 2
         1 way + 2 ways = 3 ways
Stairs:  [0]──[1]──[2]──[3]──[4]──[5]
Ways:     1    1    2    3    ?    ?

Step 3 (i=4):
To reach step 4: come from step 2 OR step 3
         2 ways + 3 ways = 5 ways
Stairs:  [0]──[1]──[2]──[3]──[4]──[5]
Ways:     1    1    2    3    5    ?

Step 4 (i=5):
To reach step 5: come from step 3 OR step 4
         3 ways + 5 ways = 8 ways
Stairs:  [0]──[1]──[2]──[3]──[4]──[5]
Ways:     1    1    2    3    5    8

All 8 ways to reach step 5:
1. 1+1+1+1+1
2. 1+1+1+2
3. 1+1+2+1
4. 1+2+1+1
5. 2+1+1+1
6. 1+2+2
7. 2+1+2
8. 2+2+1
```

---

## Concrete Example 2: 0/1 Knapsack (2D DP)

### Problem Statement

Given `n` items with weights and values, and a knapsack with capacity `W`, find the maximum value you can carry. Each item can be taken at most once.

**Input**:
- `weights = [1, 2, 3, 4]`
- `values = [10, 20, 30, 40]`
- `W = 5`

**Output**: `50` (take items with weights 1 and 4, values 10 + 40)

### Applying the 5-Step Framework

**Step 1: Define State**
```
dp[i][w] = maximum value achievable using first i items with capacity w
```

**Step 2: Base Cases**
```
dp[0][w] = 0  for all w (no items = no value)
dp[i][0] = 0  for all i (no capacity = no value)
```

**Step 3: Recurrence Relation**
```
For each item i with weight[i] and value[i]:

If weight[i] > w:  (can't take this item)
    dp[i][w] = dp[i-1][w]

Else:  (choose to take or not take)
    dp[i][w] = max(
        dp[i-1][w],                          # don't take item i
        dp[i-1][w - weight[i]] + value[i]    # take item i
    )
```

**Step 4: Iteration Order**
```
For i from 1 to n:
    For w from 1 to W:
        (row by row, left to right)
```

**Step 5: Answer Location**
```
dp[n][W] = maximum value with all items and full capacity
```

### Bottom-Up Solution

```python
def knapsack_01(weights: list[int], values: list[int], W: int) -> int:
    """
    0/1 Knapsack: Find maximum value that fits in capacity W.

    Time Complexity: O(n * W) - fill n×W table
    Space Complexity: O(n * W) - 2D dp array
    """
    n = len(weights)

    # dp[i][w] = max value using first i items with capacity w
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # Fill table
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            # Item index is i-1 (0-indexed)
            item_weight = weights[i - 1]
            item_value = values[i - 1]

            if item_weight > w:
                # Can't take this item
                dp[i][w] = dp[i - 1][w]
            else:
                # Max of: don't take vs take
                dp[i][w] = max(
                    dp[i - 1][w],                        # don't take
                    dp[i - 1][w - item_weight] + item_value  # take
                )

    return dp[n][W]
```

### Space-Optimized Solution (1D Array)

```python
def knapsack_01_optimized(weights: list[int], values: list[int], W: int) -> int:
    """
    0/1 Knapsack with O(W) space.

    Key insight: Each row only depends on the previous row.
    Trick: Iterate capacity in REVERSE to avoid overwriting needed values.

    Time Complexity: O(n * W)
    Space Complexity: O(W)
    """
    n = len(weights)
    dp = [0] * (W + 1)

    for i in range(n):
        # REVERSE iteration is crucial!
        # If we go forward, dp[w - weight[i]] would use updated (current row) value
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[W]
```

### Detailed Execution Trace

**Input**: `weights = [1, 2, 3, 4]`, `values = [10, 20, 30, 40]`, `W = 5`

**2D Table Construction:**

```
Initial table (all zeros):
              Capacity w
           0    1    2    3    4    5
        ┌────┬────┬────┬────┬────┬────┐
Item 0  │  0 │  0 │  0 │  0 │  0 │  0 │  (no items)
        ├────┼────┼────┼────┼────┼────┤
Item 1  │    │    │    │    │    │    │  weight=1, value=10
        ├────┼────┼────┼────┼────┼────┤
Item 2  │    │    │    │    │    │    │  weight=2, value=20
        ├────┼────┼────┼────┼────┼────┤
Item 3  │    │    │    │    │    │    │  weight=3, value=30
        ├────┼────┼────┼────┼────┼────┤
Item 4  │    │    │    │    │    │    │  weight=4, value=40
        └────┴────┴────┴────┴────┴────┘
```

**Row 1 (Item 1: weight=1, value=10):**

| w   | Can take? | Don't take (dp[0][w]) | Take (dp[0][w-1] + 10) | Max | dp[1][w] |
| --- | --------- | --------------------- | ---------------------- | --- | -------- |
| 1   | Yes (1≤1) | 0                     | 0 + 10 = 10            | 10  | **10**   |
| 2   | Yes (1≤2) | 0                     | 0 + 10 = 10            | 10  | **10**   |
| 3   | Yes (1≤3) | 0                     | 0 + 10 = 10            | 10  | **10**   |
| 4   | Yes (1≤4) | 0                     | 0 + 10 = 10            | 10  | **10**   |
| 5   | Yes (1≤5) | 0                     | 0 + 10 = 10            | 10  | **10**   |

**Row 2 (Item 2: weight=2, value=20):**

| w   | Can take? | Don't take | Take                     | Max | dp[2][w] |
| --- | --------- | ---------- | ------------------------ | --- | -------- |
| 1   | No (2>1)  | 10         | -                        | 10  | **10**   |
| 2   | Yes       | 10         | dp[1][0]+20 = 0+20 = 20  | 20  | **20**   |
| 3   | Yes       | 10         | dp[1][1]+20 = 10+20 = 30 | 30  | **30**   |
| 4   | Yes       | 10         | dp[1][2]+20 = 10+20 = 30 | 30  | **30**   |
| 5   | Yes       | 10         | dp[1][3]+20 = 10+20 = 30 | 30  | **30**   |

**Row 3 (Item 3: weight=3, value=30):**

| w   | Can take? | Don't take | Take                     | Max | dp[3][w] |
| --- | --------- | ---------- | ------------------------ | --- | -------- |
| 1   | No        | 10         | -                        | 10  | **10**   |
| 2   | No        | 20         | -                        | 20  | **20**   |
| 3   | Yes       | 30         | dp[2][0]+30 = 0+30 = 30  | 30  | **30**   |
| 4   | Yes       | 30         | dp[2][1]+30 = 10+30 = 40 | 40  | **40**   |
| 5   | Yes       | 30         | dp[2][2]+30 = 20+30 = 50 | 50  | **50**   |

**Row 4 (Item 4: weight=4, value=40):**

| w   | Can take? | Don't take | Take                     | Max | dp[4][w] |
| --- | --------- | ---------- | ------------------------ | --- | -------- |
| 1   | No        | 10         | -                        | 10  | **10**   |
| 2   | No        | 20         | -                        | 20  | **20**   |
| 3   | No        | 30         | -                        | 30  | **30**   |
| 4   | Yes       | 40         | dp[3][0]+40 = 0+40 = 40  | 40  | **40**   |
| 5   | Yes       | 50         | dp[3][1]+40 = 10+40 = 50 | 50  | **50**   |

**Final Table:**

```
              Capacity w
           0    1    2    3    4    5
        ┌────┬────┬────┬────┬────┬────┐
Item 0  │  0 │  0 │  0 │  0 │  0 │  0 │
        ├────┼────┼────┼────┼────┼────┤
Item 1  │  0 │ 10 │ 10 │ 10 │ 10 │ 10 │
        ├────┼────┼────┼────┼────┼────┤
Item 2  │  0 │ 10 │ 20 │ 30 │ 30 │ 30 │
        ├────┼────┼────┼────┼────┼────┤
Item 3  │  0 │ 10 │ 20 │ 30 │ 40 │ 50 │
        ├────┼────┼────┼────┼────┼────┤
Item 4  │  0 │ 10 │ 20 │ 30 │ 40 │[50]│ ← Answer
        └────┴────┴────┴────┴────┴────┘
```

**Final Output**: `dp[4][5] = 50`

**Backtracking to find items selected:**
- dp[4][5] = 50, same as dp[3][5] → didn't take item 4
- dp[3][5] = 50, different from dp[2][5]=30 → took item 3? No wait...
- Let's trace: dp[4][5]=50, dp[3][5]=50 → item 4 NOT taken
- dp[3][5]=50, dp[2][5]=30 → item 3 NOT taken (since 50=dp[2][2]+30=20+30, this means item 3 WAS taken)
- Actually: dp[3][5]=50 comes from dp[2][2]+30=50, so item 3 taken, remaining capacity=2
- dp[2][2]=20 comes from taking item 2... but that leaves capacity 0
- Let me recalculate: Take item 1 (w=1,v=10) + item 4 (w=4,v=40) = w=5, v=50 ✓

---

## Concrete Example 3: Longest Common Subsequence (String DP)

### Problem Statement

Given two strings, find the length of their longest common subsequence (LCS).

**Input**: `text1 = "abcde"`, `text2 = "ace"`
**Output**: `3` (LCS is "ace")
**Constraints**: Subsequence maintains relative order but doesn't need to be contiguous.

### Applying the 5-Step Framework

**Step 1: Define State**
```
dp[i][j] = length of LCS of text1[0:i] and text2[0:j]
```

**Step 2: Base Cases**
```
dp[0][j] = 0  for all j (empty string has no common subsequence)
dp[i][0] = 0  for all i
```

**Step 3: Recurrence Relation**
```
If text1[i-1] == text2[j-1]:  (characters match)
    dp[i][j] = dp[i-1][j-1] + 1  (extend LCS by 1)
Else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])  (skip one character)
```

**Step 4: Iteration Order**
```
Row by row, left to right (need dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
```

**Step 5: Answer Location**
```
dp[m][n] where m = len(text1), n = len(text2)
```

### Solution

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find length of longest common subsequence.

    Time Complexity: O(m * n)
    Space Complexity: O(m * n), can be optimized to O(min(m, n))
    """
    m, n = len(text1), len(text2)

    # dp[i][j] = LCS length for text1[0:i] and text2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Characters match: extend LCS
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # No match: take best of excluding one character
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def lcs_with_reconstruction(text1: str, text2: str) -> tuple[int, str]:
    """
    Find LCS length AND the actual subsequence.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find actual LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], ''.join(reversed(lcs))
```

### Detailed Execution Trace

**Input**: `text1 = "abcde"`, `text2 = "ace"`

```
          ""    a    c    e
           0    1    2    3
        ┌────┬────┬────┬────┐
""   0  │  0 │  0 │  0 │  0 │
        ├────┼────┼────┼────┤
a    1  │  0 │  1 │  1 │  1 │   text1[0]='a' matches text2[0]='a'
        ├────┼────┼────┼────┤
b    2  │  0 │  1 │  1 │  1 │   'b' doesn't match any
        ├────┼────┼────┼────┤
c    3  │  0 │  1 │  2 │  2 │   text1[2]='c' matches text2[1]='c'
        ├────┼────┼────┼────┤
d    4  │  0 │  1 │  2 │  2 │   'd' doesn't match any
        ├────┼────┼────┼────┤
e    5  │  0 │  1 │  2 │ [3]│   text1[4]='e' matches text2[2]='e'
        └────┴────┴────┴────┘
```

**Key cells explained:**
- `dp[1][1] = 1`: "a" vs "a" → match! → dp[0][0] + 1 = 1
- `dp[3][2] = 2`: "abc" vs "ac" → 'c' matches → dp[2][1] + 1 = 1 + 1 = 2
- `dp[5][3] = 3`: "abcde" vs "ace" → 'e' matches → dp[4][2] + 1 = 2 + 1 = 3

**Backtracking to find LCS "ace":**
```
Start at dp[5][3] = 3
  text1[4]='e' == text2[2]='e' → add 'e', move to dp[4][2]
At dp[4][2] = 2
  text1[3]='d' ≠ text2[1]='c'
  dp[3][2]=2 > dp[4][1]=1 → move to dp[3][2]
At dp[3][2] = 2
  text1[2]='c' == text2[1]='c' → add 'c', move to dp[2][1]
At dp[2][1] = 1
  text1[1]='b' ≠ text2[0]='a'
  dp[1][1]=1 > dp[2][0]=0 → move to dp[1][1]
At dp[1][1] = 1
  text1[0]='a' == text2[0]='a' → add 'a', move to dp[0][0]
At dp[0][0] = 0 → done

Collected (reversed): ['e', 'c', 'a'] → "ace"
```

---

## Complexity Analysis

### Time Complexity Patterns

| DP Type            | States   | Transitions | Total Time   |
| ------------------ | -------- | ----------- | ------------ |
| 1D Linear          | O(n)     | O(1)        | O(n)         |
| 1D with inner loop | O(n)     | O(n)        | O(n²)        |
| 2D Grid            | O(n × m) | O(1)        | O(n × m)     |
| 2D with choices    | O(n × m) | O(k)        | O(n × m × k) |
| Subset/Bitmask     | O(2^n)   | O(n)        | O(n × 2^n)   |

### Space Complexity Patterns

| Approach      | Space        | When Possible                                     |
| ------------- | ------------ | ------------------------------------------------- |
| Full table    | O(n × m)     | Always works                                      |
| Rolling array | O(m) or O(n) | When current row only needs previous row          |
| Two variables | O(1)         | When only need 2 previous values (Fibonacci-like) |

### Optimization Techniques

**1. Space Optimization with Rolling Array:**
```python
# Instead of dp[n][m], use dp[2][m]
# Alternate between rows: current = i % 2, previous = (i-1) % 2

def lcs_optimized(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    # Only keep 2 rows
    dp = [[0] * (n + 1) for _ in range(2)]

    for i in range(1, m + 1):
        curr = i % 2
        prev = (i - 1) % 2
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[curr][j] = dp[prev][j - 1] + 1
            else:
                dp[curr][j] = max(dp[prev][j], dp[curr][j - 1])

    return dp[m % 2][n]
```

**2. 1D Array for 0/1 Knapsack:**
```python
# Iterate capacity in REVERSE to not overwrite needed values
for i in range(n):
    for w in range(W, weights[i] - 1, -1):  # REVERSE!
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

---

## Pattern Variations

### Variation 1: Counting Problems

**When to use**: "How many ways...", "Count the number of..."
**Key difference**: Use addition instead of min/max
**Example**: Unique Paths, Coin Change 2

```python
def unique_paths(m: int, n: int) -> int:
    """Count paths from top-left to bottom-right in m×n grid."""
    dp = [[1] * n for _ in range(m)]  # First row and column are all 1s

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]  # SUM, not max

    return dp[m-1][n-1]
```

### Variation 2: Optimization Problems

**When to use**: "Find minimum/maximum..."
**Key difference**: Use min() or max() to combine subproblems
**Example**: Coin Change (min coins), Knapsack (max value)

```python
def coin_change(coins: list[int], amount: int) -> int:
    """Find minimum coins needed to make amount."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)  # MIN

    return dp[amount] if dp[amount] != float('inf') else -1
```

### Variation 3: Decision Problems

**When to use**: "Is it possible to...", "Can you..."
**Key difference**: Use boolean OR to combine subproblems
**Example**: Partition Equal Subset Sum, Word Break

```python
def can_partition(nums: list[int]) -> bool:
    """Can we partition nums into two equal-sum subsets?"""
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]  # OR

    return dp[target]
```

### Variation 4: String DP

**When to use**: Problems involving two strings
**Key difference**: 2D table where dp[i][j] relates to prefixes of both strings
**Example**: LCS, Edit Distance, Distinct Subsequences

```python
def edit_distance(word1: str, word2: str) -> int:
    """Minimum operations to convert word1 to word2."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Delete
                    dp[i][j-1],      # Insert
                    dp[i-1][j-1]     # Replace
                )

    return dp[m][n]
```

### Variation 5: Interval DP

**When to use**: Problems on contiguous ranges/intervals
**Key difference**: dp[i][j] represents optimal solution for interval [i, j]
**Example**: Matrix Chain Multiplication, Burst Balloons

```python
def matrix_chain_order(dims: list[int]) -> int:
    """
    Minimum multiplications to multiply chain of matrices.
    dims[i-1] × dims[i] is dimension of matrix i.
    """
    n = len(dims) - 1  # Number of matrices
    dp = [[0] * n for _ in range(n)]

    # length of chain
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            # Try all split points
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]
```

### Variation 6: State Machine DP

**When to use**: Problems with distinct states/modes
**Key difference**: Multiple DP arrays, one per state
**Example**: Best Time to Buy/Sell Stock with Cooldown

```python
def max_profit_with_cooldown(prices: list[int]) -> int:
    """
    Buy/sell stock with 1-day cooldown after selling.
    States: held, sold, cooldown
    """
    n = len(prices)
    if n < 2:
        return 0

    held = -prices[0]    # Max profit if holding stock
    sold = 0             # Max profit if just sold
    cooldown = 0         # Max profit if in cooldown

    for i in range(1, n):
        prev_held = held
        prev_sold = sold
        prev_cooldown = cooldown

        held = max(prev_held, prev_cooldown - prices[i])  # Hold or buy
        sold = prev_held + prices[i]                       # Sell
        cooldown = max(prev_cooldown, prev_sold)           # Stay or enter cooldown

    return max(sold, cooldown)
```

---

## Classic Problems Using This Pattern

### Problem 1: House Robber (Easy)

**Problem**: Rob houses along a street. Can't rob adjacent houses. Maximize loot.
**Key Insight**: At each house, choose to rob it (skip previous) or skip it.
**Complexity**: Time O(n), Space O(1)

```python
def rob(nums: list[int]) -> int:
    """
    dp[i] = max money robbing houses 0..i
    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2 = nums[0]             # dp[i-2]
    prev1 = max(nums[0], nums[1])  # dp[i-1]

    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current

    return prev1
```

**Edge Cases**:
- Empty array → return 0
- Single house → return that value
- Two houses → return max of both

### Problem 2: Coin Change (Medium)

**Problem**: Find minimum coins to make amount. Unlimited supply of each coin.
**Key Insight**: For each amount, try all coins and take minimum.
**Complexity**: Time O(amount × coins), Space O(amount)

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    dp[x] = minimum coins to make amount x
    dp[x] = min(dp[x], dp[x - coin] + 1) for each coin
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 0 coins to make amount 0

    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] != float('inf'):
                dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

**Edge Cases**:
- Amount 0 → return 0
- No valid combination → return -1
- Single coin equals amount → return 1

### Problem 3: Longest Increasing Subsequence (Medium)

**Problem**: Find length of longest strictly increasing subsequence.
**Key Insight**: For each element, find the best LIS ending before it that it can extend.
**Complexity**: Time O(n²) naive, O(n log n) with binary search

```python
def length_of_lis(nums: list[int]) -> int:
    """
    O(n²) solution:
    dp[i] = length of LIS ending at index i
    dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # Each element is an LIS of length 1

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def length_of_lis_optimized(nums: list[int]) -> int:
    """
    O(n log n) solution using binary search.
    Maintain array of smallest tail for LIS of each length.
    """
    from bisect import bisect_left

    tails = []  # tails[i] = smallest tail of LIS of length i+1

    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)  # Extend longest LIS
        else:
            tails[pos] = num   # Update to smaller tail

    return len(tails)
```

### Problem 4: Edit Distance (Medium)

**Problem**: Minimum operations (insert, delete, replace) to convert word1 to word2.
**Key Insight**: Classic 2D string DP. At each position, either characters match or we need an operation.
**Complexity**: Time O(m × n), Space O(m × n) or O(min(m, n))

```python
def min_distance(word1: str, word2: str) -> int:
    """
    dp[i][j] = min operations to convert word1[0:i] to word2[0:j]
    """
    m, n = len(word1), len(word2)

    # Space-optimized: only need previous row
    prev = list(range(n + 1))  # Base case: inserting j characters
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i  # Deleting i characters
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr[j] = prev[j-1]  # No operation
            else:
                curr[j] = 1 + min(
                    prev[j],      # Delete from word1
                    curr[j-1],    # Insert into word1
                    prev[j-1]     # Replace
                )
        prev, curr = curr, prev

    return prev[n]
```

### Problem 5: Partition Equal Subset Sum (Medium)

**Problem**: Can array be partitioned into two subsets with equal sum?
**Key Insight**: Reduce to: can we find subset summing to total/2? (0/1 Knapsack variant)
**Complexity**: Time O(n × sum), Space O(sum)

```python
def can_partition(nums: list[int]) -> bool:
    """
    dp[j] = True if we can make sum j using some subset
    """
    total = sum(nums)

    # Can't split odd sum equally
    if total % 2 != 0:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True  # Empty subset sums to 0

    for num in nums:
        # Iterate backwards to avoid using same element twice
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]
```

---

## Edge Cases and Gotchas

### Edge Case 1: Empty Input

**Scenario**: Array/string is empty
**Input Example**: `nums = []`
**Expected Output**: Usually 0 or problem-specific
**How to Handle**: Check at the start

```python
def solution(nums):
    if not nums:
        return 0  # or appropriate default
    # ... rest of solution
```

### Edge Case 2: Single Element

**Scenario**: Only one item in input
**Input Example**: `nums = [5]`
**How to Handle**: Often a special base case

```python
def house_robber(nums):
    if len(nums) == 1:
        return nums[0]
```

### Edge Case 3: All Same Elements

**Scenario**: All elements are identical
**Input Example**: `nums = [2, 2, 2, 2]`
**Why Tricky**: May trigger or skip certain conditions
**Test**: Ensure algorithm handles uniformity

### Edge Case 4: Integer Overflow

**Scenario**: Very large numbers or counts
**Input Example**: Counting paths in large grid
**How to Handle**: Use modular arithmetic if required

```python
MOD = 10**9 + 7
dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % MOD
```

### Edge Case 5: Negative Numbers

**Scenario**: Input contains negatives
**Input Example**: `nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]` (Max Subarray)
**How to Handle**: Ensure initialization and comparisons handle negatives

### Common Mistakes

| Mistake                    | Why It Happens                      | How to Avoid                              |
| -------------------------- | ----------------------------------- | ----------------------------------------- |
| Wrong base case            | Copying from similar problem        | Derive base case from problem definition  |
| Off-by-one errors          | Confusion between 0 and 1 indexing  | Clearly define what dp[i] means           |
| Wrong iteration order      | Dependency not understood           | Draw dependency arrows before coding      |
| Overwriting needed values  | Space optimization done incorrectly | Trace through small example               |
| Missing modular arithmetic | Forgetting when counts overflow     | Check if problem asks for answer % MOD    |
| Returning wrong cell       | dp table misunderstood              | Verify answer location matches definition |

---

## Common Misconceptions

### ❌ MISCONCEPTION: "DP is just memoized recursion"

✅ **REALITY**: Memoization is one form of DP (top-down). Tabulation (bottom-up) is equally valid and often more efficient. True DP understanding means being able to do both and knowing when each is preferable.

**WHY THE CONFUSION**: Many learn DP through recursive examples first.

**INTERVIEW TIP**: Show you can convert between approaches. Start with recursion for intuition, convert to iteration for efficiency.

### ❌ MISCONCEPTION: "If it has overlapping subproblems, use DP"

✅ **REALITY**: You need BOTH overlapping subproblems AND optimal substructure. Divide and conquer also has subproblems but they don't overlap. Greedy has optimal substructure but makes irrevocable choices.

**INTERVIEW TIP**: Explicitly identify both properties when explaining your approach.

### ❌ MISCONCEPTION: "The state is always obvious"

✅ **REALITY**: Finding the right state is often the hardest part. Wrong state leads to wrong solution or exponential complexity. Sometimes you need multiple dimensions you didn't initially consider.

**INTERVIEW TIP**: State your state definition explicitly: "Let dp[i][j] represent..."

### ❌ MISCONCEPTION: "DP always gives polynomial time"

✅ **REALITY**: DP can still be exponential if the state space is exponential (e.g., bitmask DP is O(2^n)). DP optimizes by avoiding recomputation, not by reducing state space.

**INTERVIEW TIP**: Always analyze the size of your state space.

---

## Interview Tips for This Pattern

### How to Communicate Your Approach

1. **Identify the pattern**: "This is a DP problem because I see overlapping subproblems when we... and optimal substructure since the best solution for n depends on best solutions for smaller inputs."

2. **Define the state clearly**: "Let me define dp[i] as the maximum value we can achieve considering the first i items. This captures everything we need to know to make future decisions."

3. **Explain the recurrence**: "At each step, we have two choices: take or skip. If we take, we add value[i] but must skip the adjacent, so we get dp[i-2] + value[i]. If we skip, we get dp[i-1]. We take the max."

4. **State complexity**: "We have n states, each taking O(1) to compute, giving O(n) time. We only need the last two values, so O(1) space with optimization."

5. **Handle edge cases**: "We need to handle empty input and single element separately since our recurrence assumes at least two elements."

### Common Follow-Up Questions

| Follow-Up Question                                     | What They're Testing       | How to Respond                                       |
| ------------------------------------------------------ | -------------------------- | ---------------------------------------------------- |
| "Can you optimize space?"                              | Understanding dependencies | Identify what previous states are actually needed    |
| "What if we need the actual solution, not just value?" | Reconstruction skills      | Show backtracking through DP table                   |
| "What if input is very large?"                         | Scaling awareness          | Discuss space optimization, potential approximations |
| "Is there a greedy solution?"                          | Pattern recognition        | Explain why greedy fails (counterexample) or works   |
| "What's the time complexity?"                          | Analysis skills            | Derive from # of states × work per state             |

### Red Flags to Avoid

- **Jumping to code without explaining state**: Always define dp[i] or dp[i][j] first
- **Confusing memoization and tabulation**: Know both, explain which you're using
- **Ignoring base cases**: These are crucial and interviewers check them
- **Not verifying with a small example**: Always trace through to catch errors
- **Claiming "it's just like problem X"**: Explain the actual reasoning, not just pattern matching

---

## Comparison with Related Patterns

| Aspect              | Dynamic Programming                  | Greedy                           | Divide & Conquer             |
| ------------------- | ------------------------------------ | -------------------------------- | ---------------------------- |
| Subproblem relation | Overlapping                          | None (make choice and move on)   | Non-overlapping              |
| Solution approach   | Combine optimal subproblem solutions | Make locally optimal choice      | Solve independently, combine |
| Time complexity     | Usually polynomial                   | Usually O(n log n) or O(n)       | Usually O(n log n)           |
| When to use         | Need to explore all options          | Local choice is globally optimal | Subproblems are independent  |
| Example             | Knapsack                             | Activity Selection               | Merge Sort                   |

### Decision Tree: When to Use DP vs Greedy

```
Does making a locally optimal choice always lead to global optimum?
├── Yes → Consider GREEDY
│   └── Prove with exchange argument or greedy stays ahead
└── No → Need to explore multiple options
    │
    ├── Do subproblems overlap (same subproblem solved multiple times)?
    │   ├── Yes → Use DYNAMIC PROGRAMMING
    │   └── No → Use DIVIDE AND CONQUER
    │
    └── Is there optimal substructure?
        ├── Yes → DP will work
        └── No → DP won't help, consider other approaches
```

### DP vs Greedy Example: Coin Change

**Greedy approach** (take largest coins first):
- Coins: [1, 3, 4], Amount: 6
- Greedy: 4 + 1 + 1 = 3 coins
- Optimal: 3 + 3 = 2 coins ❌ Greedy fails!

**DP approach** (consider all options):
- dp[6] = min(dp[5] + 1, dp[3] + 1, dp[2] + 1)
- Explores all possibilities, finds optimal

---

## Real-World Applications

### Application 1: Route Optimization (GPS Navigation)

**Context**: Finding shortest path in road networks
**How DP Applies**: Dijkstra's algorithm uses DP principle—shortest path to node B through A equals shortest to A plus edge A→B
**Why It Matters**: Billions of navigation queries daily rely on these algorithms

### Application 2: Text Prediction (Autocomplete)

**Context**: Predicting next word in mobile keyboards
**How DP Applies**: Hidden Markov Models and sequence prediction use DP (Viterbi algorithm)
**Why It Matters**: Improves typing speed by 30-40%

### Application 3: Resource Allocation (Cloud Computing)

**Context**: Allocating VMs to requests to minimize cost
**How DP Applies**: Variant of knapsack—maximize utilization given capacity constraints
**Why It Matters**: Saves millions in infrastructure costs

### Application 4: Bioinformatics (DNA Sequence Alignment)

**Context**: Comparing genetic sequences
**How DP Applies**: Edit distance variants (Needleman-Wunsch, Smith-Waterman algorithms)
**Why It Matters**: Foundation of genetic research, drug discovery

### Application 5: Financial Trading (Portfolio Optimization)

**Context**: Selecting investments to maximize returns
**How DP Applies**: Multi-period investment as sequential decision problem
**Why It Matters**: Manages trillions in assets globally

---

## Practice Problems (Ordered by Difficulty)

### Warm-Up (Easy)
| #   | Problem                         | Key Concept           | LeetCode # |
| --- | ------------------------------- | --------------------- | ---------- |
| 1   | Climbing Stairs                 | 1D DP, Fibonacci-like | 70         |
| 2   | Min Cost Climbing Stairs        | 1D DP with costs      | 746        |
| 3   | House Robber                    | 1D DP with constraint | 198        |
| 4   | Maximum Subarray                | Kadane's algorithm    | 53         |
| 5   | Best Time to Buy and Sell Stock | Single pass DP        | 121        |

### Core Practice (Medium)
| #   | Problem                        | Key Concept           | LeetCode # |
| --- | ------------------------------ | --------------------- | ---------- |
| 1   | Coin Change                    | Unbounded knapsack    | 322        |
| 2   | Longest Increasing Subsequence | 1D DP / Binary search | 300        |
| 3   | Unique Paths                   | 2D grid DP            | 62         |
| 4   | Minimum Path Sum               | 2D grid optimization  | 64         |
| 5   | Longest Common Subsequence     | 2D string DP          | 1143       |
| 6   | Edit Distance                  | 2D string DP          | 72         |
| 7   | Partition Equal Subset Sum     | 0/1 Knapsack variant  | 416        |
| 8   | Target Sum                     | Knapsack with +/-     | 494        |
| 9   | Word Break                     | 1D DP with dictionary | 139        |
| 10  | Decode Ways                    | 1D DP with conditions | 91         |

### Challenge (Hard)
| #   | Problem                     | Key Concept                  | LeetCode # |
| --- | --------------------------- | ---------------------------- | ---------- |
| 1   | Longest Valid Parentheses   | 1D DP with stack alternative | 32         |
| 2   | Edit Distance               | Classic string DP            | 72         |
| 3   | Regular Expression Matching | 2D DP with * handling        | 10         |
| 4   | Burst Balloons              | Interval DP                  | 312        |
| 5   | Palindrome Partitioning II  | Min cuts DP                  | 132        |
| 6   | Distinct Subsequences       | 2D counting DP               | 115        |

### Recommended Practice Order

1. **Climbing Stairs (70)** - Master the basic 1D pattern
2. **House Robber (198)** - Add the "skip adjacent" constraint
3. **Coin Change (322)** - Transition to unbounded knapsack
4. **Unique Paths (62)** - Introduction to 2D DP
5. **Longest Common Subsequence (1143)** - 2D string DP
6. **Partition Equal Subset Sum (416)** - 0/1 knapsack
7. **Longest Increasing Subsequence (300)** - O(n²) then O(n log n)
8. **Edit Distance (72)** - Classic string DP
9. **Word Break (139)** - DP with dictionary lookup
10. **Burst Balloons (312)** - Interval DP challenge

---

## Code Templates

### Template 1: 1D DP (Linear Problems)

```python
def linear_dp(nums: list[int]) -> int:
    """
    Template for 1D DP problems like House Robber, Climbing Stairs.

    Customize:
    - Base cases
    - Recurrence relation
    - What dp[i] represents
    """
    if not nums:
        return 0

    n = len(nums)

    # Option A: Full array
    dp = [0] * n
    dp[0] = nums[0]  # Base case
    # dp[1] = ...    # If needed

    for i in range(1, n):  # or range(2, n) if two base cases
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])  # Customize

    return dp[n-1]

    # Option B: Space optimized (when only need last few values)
    # prev2, prev1 = base_case_0, base_case_1
    # for i in range(2, n):
    #     curr = recurrence(prev1, prev2, nums[i])
    #     prev2, prev1 = prev1, curr
    # return prev1
```

### Template 2: 2D DP (Grid Problems)

```python
def grid_dp(grid: list[list[int]]) -> int:
    """
    Template for 2D grid problems like Unique Paths, Min Path Sum.

    Customize:
    - Base cases (first row, first column)
    - Recurrence relation
    - What dp[i][j] represents
    """
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # Base case: first cell
    dp[0][0] = grid[0][0]

    # Base case: first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]  # Customize

    # Base case: first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]  # Customize

    # Fill rest of table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]  # Customize

    return dp[m-1][n-1]
```

### Template 3: String DP (Two Strings)

```python
def string_dp(s1: str, s2: str) -> int:
    """
    Template for two-string problems like LCS, Edit Distance.

    Customize:
    - Base cases
    - Match vs no-match logic
    - What dp[i][j] represents
    """
    m, n = len(s1), len(s2)

    # dp[i][j] = answer for s1[0:i] and s2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Customize (e.g., 0 for LCS, i for edit distance)
    for j in range(n + 1):
        dp[0][j] = j  # Customize

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match: customize
            else:
                dp[i][j] = 1 + min(       # No match: customize
                    dp[i-1][j],           # Option 1
                    dp[i][j-1],           # Option 2
                    dp[i-1][j-1]          # Option 3
                )

    return dp[m][n]
```

### Template 4: Knapsack (0/1)

```python
def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Template for 0/1 Knapsack and variants.

    Customize:
    - What we're optimizing (max value, min items, etc.)
    - Constraint type
    """
    n = len(weights)

    # Space-optimized 1D array
    dp = [0] * (capacity + 1)

    for i in range(n):
        # REVERSE iteration for 0/1 (each item used once)
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


def knapsack_unbounded(weights: list[int], values: list[int], capacity: int) -> int:
    """For unbounded knapsack (items can be reused)."""
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    # OR: iterate items outside, capacity forward (not reverse)
    return dp[capacity]
```

### Template 5: Interval DP

```python
def interval_dp(nums: list[int]) -> int:
    """
    Template for interval problems like Matrix Chain, Burst Balloons.

    Key: dp[i][j] represents answer for interval [i, j]
    Iteration: by interval length, then start position
    """
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    # Base case: intervals of length 1
    for i in range(n):
        dp[i][i] = nums[i]  # Customize

    # Fill by increasing interval length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')  # or -inf for max

            # Try all split points
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + merge_cost(i, k, j)  # Customize
                dp[i][j] = min(dp[i][j], cost)  # or max

    return dp[0][n-1]
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DYNAMIC PROGRAMMING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ WHEN TO USE:                                                                │
│ • "Find minimum/maximum..." → Optimization DP                               │
│ • "Count number of ways..." → Counting DP                                   │
│ • "Is it possible to..." → Decision DP                                      │
│ • Overlapping subproblems + Optimal substructure                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ KEY IDEA: Store solutions to subproblems, combine to solve larger problems  │
├─────────────────────────────────────────────────────────────────────────────┤
│ COMPLEXITY: Time O(states × transitions) | Space O(states) or optimized    │
├─────────────────────────────────────────────────────────────────────────────┤
│ THE 5-STEP FRAMEWORK:                                                       │
│   1. Define state: What does dp[i] or dp[i][j] represent?                   │
│   2. Base cases: What's known without computation?                          │
│   3. Recurrence: How does current state depend on previous?                 │
│   4. Order: Which direction to fill the table?                              │
│   5. Answer: Where in the table is the final answer?                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ TWO APPROACHES:                                                             │
│   Top-Down (Memoization): Recursive + cache, natural thinking               │
│   Bottom-Up (Tabulation): Iterative, often faster, easier space opt         │
├─────────────────────────────────────────────────────────────────────────────┤
│ SPACE OPTIMIZATION:                                                         │
│   • Rolling array: O(n) → O(1) when only need previous row                  │
│   • 0/1 Knapsack: iterate capacity in REVERSE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ WATCH OUT FOR:                                                              │
│ • Wrong base cases (derive from problem, don't copy)                        │
│ • Off-by-one errors (clearly define what dp[i] means)                       │
│ • Wrong iteration order (ensure dependencies computed first)                │
│ • Integer overflow (use MOD if needed)                                      │
│ • Returning wrong cell (verify answer location)                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Connections to Other Patterns

### Prerequisites
- **Recursion** - DP is optimized recursion; understand call stack and base cases
- **Arrays/Strings** - Most DP operates on sequential data
- **Basic Math** - Recurrence relations, combinatorics for counting problems

### Builds Upon
- Recursion with memoization is the bridge to DP
- Divide and conquer when subproblems don't overlap
- Brute force enumeration (DP optimizes this)

### Leads To
- **Advanced DP**: Bitmask DP, Digit DP, DP on Trees
- **Optimization algorithms**: Linear programming connections
- **Graph algorithms**: Many use DP principles (shortest path, etc.)

### Often Combined With
- **Binary Search + DP**: LIS in O(n log n)
- **Greedy + DP**: Sometimes hybrid approaches work
- **Graph + DP**: Shortest paths, DAG problems
- **Bit Manipulation + DP**: Subset-based problems

---

## Summary: Key Takeaways

1. **Core Principle**: Break problem into overlapping subproblems, solve each once, combine for final answer.

2. **When to Use**: Problems asking for min/max/count where current choice affects future options, and same subproblems are solved repeatedly.

3. **Key Insight**: The state definition is everything—wrong state means wrong solution. Always explicitly define what dp[i] represents.

4. **Common Mistake**: Jumping to code without understanding the recurrence. Always derive the recurrence relation first, verify with small example.

5. **Interview Tip**: Start by identifying the two key properties (overlapping subproblems + optimal substructure), clearly state your state definition, write the recurrence, then code. Mention space optimization as a follow-up.

---

## Additional Resources

### Video Explanations
- **MIT OpenCourseWare 6.006** - Excellent theoretical foundation
- **Back to Back SWE** - Visual, intuitive explanations
- **NeetCode** - Problem-focused, interview-oriented

### Reading
- **Introduction to Algorithms (CLRS)** - Chapter 15, rigorous treatment
- **Grokking Dynamic Programming Patterns** - Pattern-based approach
- **Elements of Programming Interviews** - Interview-focused problems

### Interactive Practice
- **LeetCode** - Filter by "Dynamic Programming" tag, sort by acceptance rate
- **AtCoder Educational DP Contest** - 26 classic DP problems
- **Codeforces** - Search for "dp" tag, filter by difficulty
