# Module 16: Greedy Algorithms

## Pattern Overview

### The Core Problem This Pattern Solves

Greedy algorithms solve optimization problems by making the **locally optimal choice at each step**, hoping this leads to a **globally optimal solution**. When greedy works, it's often the simplest and most efficient approach.

**Imagine trying to make change for 67 cents with US coins (25¢, 10¢, 5¢, 1¢):**
- Greedy: Take the largest coin that fits, repeat
- 25¢ → 25¢ → 10¢ → 5¢ → 1¢ → 1¢ = 6 coins ✓ (This is optimal!)
- No need to explore all combinations

**What types of problems become solvable with Greedy?**
- Scheduling and interval problems
- Graph problems (MST, shortest path with non-negative weights)
- Huffman coding and data compression
- Fractional optimization (fractional knapsack)
- Problems with "greedy choice property"

**The brute force approach (why Greedy is better when it works):**
- Enumerate all possibilities: O(2^n) or O(n!)
- Dynamic Programming: O(n²) or O(n × W)
- Greedy: Often O(n log n) or O(n)

**The catch:** Greedy doesn't always work! You must prove correctness.

### When to Recognize This Pattern

**Problem Signals** (keywords and characteristics):
- **"Find minimum/maximum number of..."**: Minimum coins, maximum activities
- **"Schedule/select to maximize/minimize..."**: Meeting rooms, job scheduling
- **"Optimal way to partition/distribute..."**: Load balancing, task assignment
- **"Earliest/latest/largest/smallest first..."**: Hints at sorting strategy
- **Problems where local decisions don't affect future options' optimality**

**Input Characteristics**:
- Items can be sorted by some criteria (deadline, ratio, end time)
- Making a choice eliminates some options but doesn't change remaining options' values
- Problem has "greedy choice property" and "optimal substructure"

**Two Key Properties (MUST have both for Greedy to work):**

| Property                   | Definition                                                                      | How to Verify                                                                      |
| -------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Greedy Choice Property** | A globally optimal solution can be arrived at by making locally optimal choices | Prove: optimal solution can be modified to include greedy choice without worsening |
| **Optimal Substructure**   | Optimal solution contains optimal solutions to subproblems                      | After greedy choice, remaining problem is smaller instance of same problem         |

### Real-World Analogy

**Packing a Suitcase with Weight Limit (Fractional Knapsack)**

You're packing for a trip with a weight limit. You have items of different weights and values (say, value per kg).

- **Greedy approach**: Always pack items with highest value-per-kg first
- If the best item doesn't fit entirely, take as much as you can
- This is provably optimal for fractional knapsack!

**What the analogy captures well:**
- Making the "obviously best" choice at each step
- Not needing to reconsider past decisions
- Sorting by some efficiency metric (value/weight ratio)

**Where the analogy breaks down:**
- In 0/1 knapsack (can't take fractions), greedy fails
- Example: Items [(weight=10, value=60), (weight=20, value=100), (weight=30, value=120)], capacity=50
- Greedy by ratio: Take item 1 (ratio=6), then item 2 (ratio=5) → value=160
- Optimal: Take items 2 and 3 → value=220 ❌ Greedy fails!

---

## Key Concepts at a Glance

| Term                       | Definition                                                               | Why It Matters                           |
| -------------------------- | ------------------------------------------------------------------------ | ---------------------------------------- |
| **Greedy Choice**          | The locally optimal decision at current step                             | Must prove this leads to global optimum  |
| **Greedy Choice Property** | Local optimum leads to global optimum                                    | Without this, greedy gives wrong answer  |
| **Exchange Argument**      | Proof technique: swap non-greedy choice with greedy, show no worse       | Standard way to prove greedy correctness |
| **Greedy Stays Ahead**     | Proof technique: show greedy is always ≥ any other solution at each step | Alternative proof method                 |
| **Sorting Criterion**      | The metric by which we order choices                                     | Wrong criterion = wrong answer           |
| **Irrevocable Choice**     | Once made, greedy choices aren't reconsidered                            | Key difference from DP/backtracking      |

---

## The Pattern Mechanics

### Core Idea in One Sentence

**Sort candidates by some criterion, then iteratively select the best available option that satisfies constraints, never looking back.**

### Greedy vs Other Paradigms

```
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│       GREEDY            │  │  DYNAMIC PROGRAMMING    │  │     BACKTRACKING        │
├─────────────────────────┤  ├─────────────────────────┤  ├─────────────────────────┤
│                         │  │                         │  │                         │
│  Make best local choice │  │   Explore all choices   │  │     Try a choice        │
│           ↓             │  │           ↓             │  │           ↓             │
│  Move to smaller problem│  │  Store all subproblem   │  │        Recurse          │
│           ↓             │  │       results           │  │           ↓             │
│    Never reconsider     │  │           ↓             │  │      Undo if bad        │
│                         │  │   Combine for optimal   │  │           ↓             │
│                         │  │                         │  │    (Back to choice)     │
│                         │  │                         │  │                         │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

### The Greedy Algorithm Template

```
                    1. SORT candidates by greedy criterion
                                    ↓
                    2. INITIALIZE result set/counter
                                    ↓
                         ┌──────────────────┐
                         │ 3. More          │
                    ┌────│    candidates?   │────┐
                    │    └──────────────────┘    │
                   Yes                           No
                    │                             │
                    ↓                             ↓
          ┌─────────────────────┐        7. RETURN result
          │ 4. Can select this  │
          │    candidate?       │
          └─────────────────────┘
                    │
            ┌───────┴───────┐
           Yes             No
            │               │
            ↓               ↓
    5. SELECT:        6. SKIP:
    Add to result,    Move to next
    update state
            │               │
            └───────┬───────┘
                    │
                    ↓
            (Back to step 3)
```

### Proof Techniques

```
EXCHANGE ARGUMENT                      GREEDY STAYS AHEAD
══════════════════                     ══════════════════

  Assume optimal solution O              After step k
            ↓                                  ↓
  O differs from greedy                 Greedy has made k choices
     at some point                             ↓
            ↓                            Any other approach has
  Swap: Replace O's choice                 ≤ k choices
     with greedy's                             ↓
            ↓                            Or greedy's k choices
  Show: New solution                        are 'better'
     is no worse                               ↓
            ↓                            By induction: greedy wins
  Conclude: Greedy choice
     is safe
```

### Why Greedy Works (When It Does)

**The key insight**: If making the locally optimal choice never "blocks" us from reaching the global optimum, greedy works.

**Example: Activity Selection**
- Sort activities by end time
- Always pick the earliest-ending activity that doesn't conflict
- Why it works: Picking earliest-ending leaves maximum room for future activities
- Any other choice can only leave less room (or equal)

**When Greedy Fails: 0/1 Knapsack**
- Picking the best value/weight ratio might use up capacity suboptimally
- A lower-ratio item might "fit better" with others
- Need DP to explore all combinations

---

## Concrete Example 1: Activity Selection

### Problem Statement

Given `n` activities with start and end times, find the maximum number of non-overlapping activities you can perform.

**Input**:
```
activities = [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]
             (start, end) pairs
```

**Output**: `4` (activities at indices 0, 3, 7, 10 → times (1,4), (5,7), (8,11), (12,16))

### Applying Greedy

**Greedy Strategy**: Sort by end time, always pick the earliest-ending non-conflicting activity.

**Why this criterion?**
- Picking the activity that ends earliest leaves maximum time for remaining activities
- Any other choice ends same time or later → can only fit ≤ activities

```
         Sort activities by END TIME
                    ↓
         Select first activity
                    ↓
         For each remaining activity
                    ↓
         ┌──────────────────────────┐
         │ Starts after last        │
         │ selected ends?           │
         └──────────────────────────┘
                    │
         ┌──────────┴──────────┐
        Yes                    No
         │                      │
         ↓                      ↓
   SELECT this            SKIP this
   activity               activity
         │                      │
         ↓                      │
   Update:                     │
   last_end = this.end         │
         │                      │
         └──────────┬───────────┘
                    │
                    ↓
         (Back to next activity)
                    ↓
         Return count of selected
```

### Solution

```python
def activity_selection(activities: list[tuple[int, int]]) -> list[int]:
    """
    Select maximum number of non-overlapping activities.

    Args:
        activities: List of (start, end) tuples

    Returns:
        List of indices of selected activities

    Time Complexity: O(n log n) for sorting
    Space Complexity: O(n) for storing result
    """
    if not activities:
        return []

    # Create list of (end, start, original_index) and sort by end time
    indexed = [(end, start, i) for i, (start, end) in enumerate(activities)]
    indexed.sort()  # Sorts by end time (first element)

    selected = [indexed[0][2]]  # Select first activity (earliest end)
    last_end = indexed[0][0]

    for end, start, idx in indexed[1:]:
        if start >= last_end:  # Non-overlapping
            selected.append(idx)
            last_end = end

    return selected


def activity_selection_count(activities: list[tuple[int, int]]) -> int:
    """Return just the count of maximum activities."""
    if not activities:
        return 0

    # Sort by end time
    sorted_acts = sorted(activities, key=lambda x: x[1])

    count = 1
    last_end = sorted_acts[0][1]

    for start, end in sorted_acts[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

### Detailed Execution Trace

**Input** (sorted by end time):
```
Index: 0:(1,4), 1:(3,5), 2:(0,6), 3:(5,7), 4:(3,9), 5:(5,9), 6:(6,10), 7:(8,11), 8:(8,12), 9:(2,14), 10:(12,16)
```

| Step | Activity | Start | End | last_end | Condition | Action | Selected                        |
| ---- | -------- | ----- | --- | -------- | --------- | ------ | ------------------------------- |
| 0    | (1,4)    | 1     | 4   | -        | First     | SELECT | [(1,4)]                         |
| 1    | (3,5)    | 3     | 5   | 4        | 3 < 4     | SKIP   | [(1,4)]                         |
| 2    | (0,6)    | 0     | 6   | 4        | 0 < 4     | SKIP   | [(1,4)]                         |
| 3    | (5,7)    | 5     | 7   | 4        | 5 ≥ 4 ✓   | SELECT | [(1,4), (5,7)]                  |
| 4    | (3,9)    | 3     | 9   | 7        | 3 < 7     | SKIP   | [(1,4), (5,7)]                  |
| 5    | (5,9)    | 5     | 9   | 7        | 5 < 7     | SKIP   | [(1,4), (5,7)]                  |
| 6    | (6,10)   | 6     | 10  | 7        | 6 < 7     | SKIP   | [(1,4), (5,7)]                  |
| 7    | (8,11)   | 8     | 11  | 7        | 8 ≥ 7 ✓   | SELECT | [(1,4), (5,7), (8,11)]          |
| 8    | (8,12)   | 8     | 12  | 11       | 8 < 11    | SKIP   | [(1,4), (5,7), (8,11)]          |
| 9    | (2,14)   | 2     | 14  | 11       | 2 < 11    | SKIP   | [(1,4), (5,7), (8,11)]          |
| 10   | (12,16)  | 12    | 16  | 11       | 12 ≥ 11 ✓ | SELECT | [(1,4), (5,7), (8,11), (12,16)] |

**Final Output**: 4 activities selected

### Visual Timeline

```
Activity Selection - Greedy Solution
Time: 0    2    4    6    8    10   12   14   16
      |----|----|----|----|----|----|----|----|----|

SELECTED (✓):
Activity 0:  [====]                                      (1-4)
Activity 3:            [===]                             (5-7)
Activity 7:                      [====]                  (8-11)
Activity 10:                               [======]      (12-16)

SKIPPED (✗):
Activity 2: [========]                                   (0-6)
Activity 9:    [=========================]               (2-14)
Activity 1:      [===]                                   (3-5)
Activity 4:      [========]                              (3-9)
Activity 5:            [======]                          (5-9)
Activity 6:               [======]                       (6-10)
Activity 8:                      [======]                (8-12)
```

### Proof of Correctness (Exchange Argument)

**Claim**: Greedy (select by earliest end time) gives optimal solution.

**Proof**:
1. Let OPT be an optimal solution
2. Let G be our greedy solution
3. If G = OPT, we're done
4. Otherwise, let `i` be the first position where they differ
   - G selected activity `g` with end time `end_g`
   - OPT selected activity `o` with end time `end_o`
   - Since greedy picks earliest end: `end_g ≤ end_o`
5. **Exchange**: Replace `o` with `g` in OPT
   - This is valid because `g` ends no later than `o`
   - All activities after position `i` in OPT still fit
6. New solution OPT' has same size as OPT but matches G at position `i`
7. Repeat until OPT = G
8. Therefore, G is optimal ∎

---

## Concrete Example 2: Fractional Knapsack

### Problem Statement

Given items with weights and values, and a knapsack with capacity `W`, maximize value. **You can take fractions of items.**

**Input**:
```
items = [(weight=10, value=60), (weight=20, value=100), (weight=30, value=120)]
W = 50
```

**Output**: `240` (take all of item 0, all of item 1, and 2/3 of item 2)

### Applying Greedy

**Greedy Strategy**: Sort by value/weight ratio (descending), take items greedily.

**Why this works for fractional but not 0/1?**
- Fractional: Always taking highest ratio is optimal (can adjust amounts)
- 0/1: Can't adjust, might need suboptimal ratio item to fill capacity better

```
         Calculate value/weight RATIO for each item
                            ↓
         Sort items by ratio DESCENDING
                            ↓
            remaining_capacity = W
                            ↓
         ┌──────────────────────────────┐
         │ More items AND               │
    ┌────│ capacity > 0?                │────┐
    │    └──────────────────────────────┘    │
   Yes                                       No
    │                                         │
    ↓                                         ↓
    ┌──────────────────────┐          Return total value
    │ item.weight ≤        │
    │ remaining?           │
    └──────────────────────┘
              │
      ┌───────┴───────┐
     Yes             No
      │               │
      ↓               ↓
 Take ENTIRE      Take FRACTION
    item            of item
      │               │
      ↓               ↓
 value += item    fraction = remaining / item.weight
 .value           value += fraction * item.value
 remaining -=     remaining = 0
 item.weight
      │               │
      └───────┬───────┘
              │
              ↓
    (Back to check more items)
```

### Solution

```python
def fractional_knapsack(items: list[tuple[int, int]], capacity: int) -> float:
    """
    Fractional Knapsack: Maximize value with fractional items allowed.

    Args:
        items: List of (weight, value) tuples
        capacity: Maximum weight capacity

    Returns:
        Maximum value achievable

    Time Complexity: O(n log n) for sorting
    Space Complexity: O(n) for ratio list
    """
    if not items or capacity <= 0:
        return 0.0

    # Calculate (ratio, weight, value) and sort by ratio descending
    ratios = [(value / weight, weight, value) for weight, value in items]
    ratios.sort(reverse=True)  # Highest ratio first

    total_value = 0.0
    remaining = capacity

    for ratio, weight, value in ratios:
        if remaining <= 0:
            break

        if weight <= remaining:
            # Take entire item
            total_value += value
            remaining -= weight
        else:
            # Take fraction of item
            fraction = remaining / weight
            total_value += fraction * value
            remaining = 0

    return total_value


def fractional_knapsack_detailed(items: list[tuple[int, int]], capacity: int) -> tuple[float, list]:
    """Returns value and details of what was taken."""
    if not items or capacity <= 0:
        return 0.0, []

    # (ratio, weight, value, original_index)
    ratios = [(value / weight, weight, value, i) for i, (weight, value) in enumerate(items)]
    ratios.sort(reverse=True)

    total_value = 0.0
    remaining = capacity
    taken = []  # (index, fraction_taken, value_gained)

    for ratio, weight, value, idx in ratios:
        if remaining <= 0:
            break

        if weight <= remaining:
            taken.append((idx, 1.0, value))
            total_value += value
            remaining -= weight
        else:
            fraction = remaining / weight
            value_gained = fraction * value
            taken.append((idx, fraction, value_gained))
            total_value += value_gained
            remaining = 0

    return total_value, taken
```

### Detailed Execution Trace

**Input**: `items = [(10, 60), (20, 100), (30, 120)]`, `W = 50`

**Step 1: Calculate ratios**
| Item | Weight | Value | Ratio (V/W) |
| ---- | ------ | ----- | ----------- |
| 0    | 10     | 60    | 6.0         |
| 1    | 20     | 100   | 5.0         |
| 2    | 30     | 120   | 4.0         |

**Step 2: Sort by ratio (descending)**: Order is [0, 1, 2]

**Step 3: Greedy selection**

| Step | Item | Ratio | Weight | Remaining | Action             | Value Added | Total Value |
| ---- | ---- | ----- | ------ | --------- | ------------------ | ----------- | ----------- |
| 1    | 0    | 6.0   | 10     | 50        | Take all (10 ≤ 50) | 60          | 60          |
| 2    | 1    | 5.0   | 20     | 40        | Take all (20 ≤ 40) | 100         | 160         |
| 3    | 2    | 4.0   | 30     | 20        | Take 20/30 = 2/3   | 80          | **240**     |

**Final Output**: `240.0`

### Visual Representation

```
Knapsack Contents (Capacity = 50)
═══════════════════════════════════

Item 0 (100%):  [##########]                    Weight: 10 / 10 (Full)
Item 1 (100%):  [####################]           Weight: 20 / 20 (Full)
Item 2 (67%):   [#############·······]          Weight: 20 / 30 (2/3)
                ├────────────────────────────┤
                0         25                 50

Total Weight: 50 / 50 (Full capacity)
Total Value: 240
```

### Why Greedy Works Here (Proof Sketch)

**Claim**: Taking items by highest ratio is optimal for fractional knapsack.

**Proof**:
1. Suppose optimal solution OPT takes less of highest-ratio item than greedy
2. OPT must take more of some lower-ratio item to fill capacity
3. **Exchange**: Replace some lower-ratio amount with higher-ratio amount
4. Same weight, but higher value (since ratio is higher)
5. This contradicts OPT being optimal
6. Therefore, greedy is optimal ∎

---

## Concrete Example 3: Huffman Coding

### Problem Statement

Given characters with frequencies, build an optimal prefix-free binary encoding that minimizes total encoded length.

**Input**:
```
chars = ['a', 'b', 'c', 'd', 'e', 'f']
freqs = [5, 9, 12, 13, 16, 45]
```

**Output**: Variable-length codes where frequent characters have shorter codes.

### Applying Greedy

**Greedy Strategy**: Repeatedly combine the two lowest-frequency nodes.

**Why this works?**
- Lowest frequency characters should have longest codes (deep in tree)
- By combining lowest two, we ensure they become siblings at deepest level
- This minimizes weighted path length

```
      Create leaf node for each character
                    ↓
      Add all nodes to min-heap by frequency
                    ↓
              ┌─────────────┐
              │ Heap has    │
         ┌────│ > 1 node?   │────┐
         │    └─────────────┘    │
        Yes                      No
         │                        │
         ↓                        ↓
   Extract two            Remaining node is root
   minimum nodes                  ↓
         │                 Traverse tree:
         ↓                 left=0, right=1
   Create parent with
   freq = sum
         │
         ↓
   Add parent back to heap
         │
         └────────┐
                  │
                  ↓
          (Back to check heap)
```

### Solution

```python
import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char: str | None, freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_coding(chars: list[str], freqs: list[int]) -> dict[str, str]:
    """
    Build Huffman codes for characters based on frequencies.

    Args:
        chars: List of characters
        freqs: Corresponding frequencies

    Returns:
        Dictionary mapping characters to binary codes

    Time Complexity: O(n log n) - n heap operations
    Space Complexity: O(n) - tree nodes
    """
    if len(chars) == 1:
        return {chars[0]: '0'}

    # Create leaf nodes and add to min-heap
    heap = [HuffmanNode(c, f) for c, f in zip(chars, freqs)]
    heapq.heapify(heap)

    # Build tree by combining two smallest nodes
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Create internal node with combined frequency
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right

        heapq.heappush(heap, parent)

    # Generate codes by traversing tree
    root = heap[0]
    codes = {}

    def generate_codes(node: HuffmanNode, code: str):
        if node.char is not None:  # Leaf node
            codes[node.char] = code if code else '0'
            return
        if node.left:
            generate_codes(node.left, code + '0')
        if node.right:
            generate_codes(node.right, code + '1')

    generate_codes(root, '')
    return codes


def huffman_encode(text: str, codes: dict[str, str]) -> str:
    """Encode text using Huffman codes."""
    return ''.join(codes[c] for c in text)


def calculate_compression(chars: list[str], freqs: list[int], codes: dict[str, str]) -> dict:
    """Calculate compression statistics."""
    # Fixed-length encoding would need ceil(log2(n)) bits per char
    import math
    fixed_bits = math.ceil(math.log2(len(chars))) if len(chars) > 1 else 1

    total_chars = sum(freqs)
    fixed_total = total_chars * fixed_bits
    huffman_total = sum(freqs[i] * len(codes[chars[i]]) for i in range(len(chars)))

    return {
        'fixed_bits_per_char': fixed_bits,
        'fixed_total_bits': fixed_total,
        'huffman_total_bits': huffman_total,
        'compression_ratio': huffman_total / fixed_total,
        'space_saved': f"{(1 - huffman_total/fixed_total) * 100:.1f}%"
    }
```

### Detailed Execution Trace

**Input**: `chars = ['a', 'b', 'c', 'd', 'e', 'f']`, `freqs = [5, 9, 12, 13, 16, 45]`

**Building the Huffman Tree:**

| Step | Heap State (freq)                  | Extract        | New Node    | Action             |
| ---- | ---------------------------------- | -------------- | ----------- | ------------------ |
| 0    | [5:a, 9:b, 12:c, 13:d, 16:e, 45:f] | -              | -           | Initial            |
| 1    | [12:c, 13:d, 14:ab, 16:e, 45:f]    | 5:a, 9:b       | 14:(a,b)    | Combine a+b        |
| 2    | [14:ab, 16:e, 25:cd, 45:f]         | 12:c, 13:d     | 25:(c,d)    | Combine c+d        |
| 3    | [25:cd, 30:abe, 45:f]              | 14:ab, 16:e    | 30:(ab,e)   | Combine (ab)+e     |
| 4    | [45:f, 55:cdabe]                   | 25:cd, 30:abe  | 55:(cd,abe) | Combine (cd)+(abe) |
| 5    | [100:root]                         | 45:f, 55:cdabe | 100:root    | Final tree         |

### Visual: Huffman Tree

```
                           Root (100)
                          /         \
                        0/           \1
                        /             \
                    f:45               (55)
                                      /    \
                                    0/      \1
                                    /        \
                                  (25)       (30)
                                 /   \       /   \
                               0/     \1   0/     \1
                               /       \   /       \
                            c:12     d:13 (14)    e:16
                                         /  \
                                       0/    \1
                                       /      \
                                     a:5     b:9

Legend: 0 = left branch, 1 = right branch
Codes: f=0, c=100, d=101, a=1100, b=1101, e=111
```

**Generated Codes** (left=0, right=1):

| Character | Frequency | Code | Code Length | Bits Used |
| --------- | --------- | ---- | ----------- | --------- |
| f         | 45        | 0    | 1           | 45        |
| c         | 12        | 100  | 3           | 36        |
| d         | 13        | 101  | 3           | 39        |
| e         | 16        | 111  | 3           | 48        |
| a         | 5         | 1100 | 4           | 20        |
| b         | 9         | 1101 | 4           | 36        |

**Total Huffman bits**: 45 + 36 + 39 + 48 + 20 + 36 = **224 bits**
**Fixed-length (3 bits each)**: 100 × 3 = **300 bits**
**Compression**: 224/300 = 74.7% (saved 25.3%)

---

## Complexity Analysis

### Time Complexity Patterns

| Problem Type          | Sorting      | Selection           | Total          |
| --------------------- | ------------ | ------------------- | -------------- |
| Activity Selection    | O(n log n)   | O(n)                | O(n log n)     |
| Fractional Knapsack   | O(n log n)   | O(n)                | O(n log n)     |
| Huffman Coding        | O(n) heapify | O(n log n) heap ops | O(n log n)     |
| Minimum Spanning Tree | O(E log E)   | O(E α(V))           | O(E log E)     |
| Dijkstra's Algorithm  | -            | O((V+E) log V)      | O((V+E) log V) |

### Space Complexity Patterns

| Approach         | Space            | Notes                      |
| ---------------- | ---------------- | -------------------------- |
| In-place greedy  | O(1)             | Modify input, track result |
| With sorting     | O(n) or O(log n) | Depends on sort algorithm  |
| With heap        | O(n)             | Heap storage               |
| With result list | O(k)             | k = size of result         |

### Why Greedy is Often Efficient

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Greedy: O(n log n)  │  │  DP: O(n × W) or     │  │ Brute Force: O(2ⁿ)  │
│                      │  │      O(n²)           │  │                      │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│  Sort once           │  │  Build table         │  │  Try all subsets     │
│      ↓               │  │      ↓               │  │                      │
│  Single pass         │  │  Fill all cells      │  │  (Exponential time)  │
│                      │  │                      │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

---

## Pattern Variations

### Variation 1: Interval Scheduling Variants

**Maximum Non-Overlapping (Activity Selection)**
```python
def max_non_overlapping(intervals):
    """Sort by end time, greedily select."""
    intervals.sort(key=lambda x: x[1])
    count, last_end = 0, float('-inf')
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    return count
```

**Minimum Rooms (Meeting Rooms II)**
```python
def min_meeting_rooms(intervals):
    """Sort events, track concurrent meetings."""
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends
    events.sort()

    rooms = max_rooms = 0
    for time, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)
    return max_rooms
```

**Minimum Intervals to Remove (Non-overlapping Intervals)**
```python
def erase_overlap_intervals(intervals):
    """Count overlapping intervals to remove."""
    intervals.sort(key=lambda x: x[1])
    count, last_end = 0, float('-inf')
    for start, end in intervals:
        if start >= last_end:
            last_end = end
        else:
            count += 1  # Remove this interval
    return count
```

### Variation 2: Jump/Reach Problems

**Jump Game (Can Reach End?)**
```python
def can_jump(nums: list[int]) -> bool:
    """Track farthest reachable position."""
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest:
            return False
        farthest = max(farthest, i + jump)
    return farthest >= len(nums) - 1
```

**Jump Game II (Minimum Jumps)**
```python
def min_jumps(nums: list[int]) -> int:
    """BFS-like: process level by level."""
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
            if current_end >= len(nums) - 1:
                break

    return jumps
```

### Variation 3: Partition/Distribution Problems

**Partition Labels**
```python
def partition_labels(s: str) -> list[int]:
    """Each letter appears in at most one part."""
    last = {c: i for i, c in enumerate(s)}  # Last occurrence

    partitions = []
    start = end = 0

    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            partitions.append(end - start + 1)
            start = i + 1

    return partitions
```

### Variation 4: Greedy with Sorting by Custom Criteria

**Task Scheduler**
```python
def least_interval(tasks: list[str], n: int) -> int:
    """Minimum intervals to complete all tasks with cooldown n."""
    from collections import Counter

    freq = list(Counter(tasks).values())
    max_freq = max(freq)
    max_count = freq.count(max_freq)

    # Formula: (max_freq - 1) * (n + 1) + max_count
    # Or just len(tasks) if we have enough variety
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
```

**Reorganize String**
```python
def reorganize_string(s: str) -> str:
    """Rearrange so no two adjacent chars are same."""
    from collections import Counter
    import heapq

    freq = Counter(s)
    max_freq = max(freq.values())

    # Impossible if any char > (n+1)/2
    if max_freq > (len(s) + 1) // 2:
        return ""

    # Use max-heap, alternate between most frequent
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)

    result = []
    prev_count, prev_char = 0, ''

    while heap:
        count, char = heapq.heappop(heap)
        result.append(char)

        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))

        prev_count, prev_char = count + 1, char

    return ''.join(result)
```

---

## Classic Problems Using This Pattern

### Problem 1: Gas Station (Medium)

**Problem**: Circular route with gas stations. Find starting station to complete circuit, or -1 if impossible.
**Key Insight**: If total gas ≥ total cost, solution exists. Start after the point with lowest cumulative sum.
**Complexity**: Time O(n), Space O(1)

```python
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """
    Greedy: Track cumulative sum, restart when negative.
    If total gas >= total cost, solution exists.
    """
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        if current_tank < 0:
            # Can't reach i+1 from start, try starting at i+1
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1
```

### Problem 2: Candy Distribution (Hard)

**Problem**: Each child has a rating. Give candies such that: (1) each child gets ≥1, (2) higher-rated children get more than neighbors.
**Key Insight**: Two passes - left-to-right for right neighbor, right-to-left for left neighbor.
**Complexity**: Time O(n), Space O(n)

```python
def candy(ratings: list[int]) -> int:
    """
    Two-pass greedy: satisfy left neighbor, then right neighbor.
    """
    n = len(ratings)
    candies = [1] * n

    # Left to right: if rating[i] > rating[i-1], give more than left neighbor
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Right to left: if rating[i] > rating[i+1], may need more than right neighbor
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)
```

### Problem 3: Assign Cookies (Easy)

**Problem**: Assign cookies to children. Child `i` is satisfied if cookie size ≥ greed[i]. Maximize satisfied children.
**Key Insight**: Sort both, greedily match smallest sufficient cookie to least greedy child.
**Complexity**: Time O(n log n), Space O(1)

```python
def find_content_children(greed: list[int], cookies: list[int]) -> int:
    """
    Sort both arrays, greedily assign smallest sufficient cookie.
    """
    greed.sort()
    cookies.sort()

    child = cookie = 0
    while child < len(greed) and cookie < len(cookies):
        if cookies[cookie] >= greed[child]:
            child += 1  # Child satisfied
        cookie += 1  # Try next cookie either way

    return child
```

### Problem 4: Minimum Number of Arrows to Burst Balloons (Medium)

**Problem**: Balloons at [start, end] intervals. Arrow at x bursts all balloons where start ≤ x ≤ end. Find minimum arrows.
**Key Insight**: Same as activity selection - sort by end, count non-overlapping groups.
**Complexity**: Time O(n log n), Space O(1)

```python
def find_min_arrow_shots(points: list[list[int]]) -> int:
    """
    Equivalent to counting non-overlapping interval groups.
    Sort by end, shoot at end of first balloon in each group.
    """
    if not points:
        return 0

    points.sort(key=lambda x: x[1])
    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows
```

### Problem 5: Queue Reconstruction by Height (Medium)

**Problem**: People with (height, k) where k = number of people in front with height ≥. Reconstruct the queue.
**Key Insight**: Sort by height descending, k ascending. Insert at index k.
**Complexity**: Time O(n²), Space O(n)

```python
def reconstruct_queue(people: list[list[int]]) -> list[list[int]]:
    """
    Greedy: Process tallest first. Insert at position k.
    Shorter people don't affect k of taller people.
    """
    # Sort: tallest first, then by k (smaller k first)
    people.sort(key=lambda x: (-x[0], x[1]))

    queue = []
    for person in people:
        queue.insert(person[1], person)

    return queue
```

---

## Edge Cases and Gotchas

### Edge Case 1: Empty Input

**Scenario**: No items/intervals to process
**How to Handle**: Return 0 or empty result immediately

```python
def greedy_solution(items):
    if not items:
        return 0  # or []
```

### Edge Case 2: Single Element

**Scenario**: Only one item
**How to Handle**: Often a simple return

```python
def activity_selection(activities):
    if len(activities) == 1:
        return 1  # Can always do one activity
```

### Edge Case 3: All Same Values

**Scenario**: All items have identical priority/weight
**How to Handle**: Algorithm should still work, but verify

### Edge Case 4: Floating Point Precision

**Scenario**: Ratios or fractions lead to precision issues
**How to Handle**: Use integer arithmetic when possible, or tolerance comparison

```python
# Instead of ratio comparison
# Use cross multiplication: a/b > c/d ⟺ a*d > b*c
def compare_ratios(v1, w1, v2, w2):
    return v1 * w2 > v2 * w1  # Avoids division
```

### Common Mistakes

| Mistake                 | Why It Happens                    | How to Avoid                             |
| ----------------------- | --------------------------------- | ---------------------------------------- |
| Wrong sorting criterion | Intuition misleads                | Prove with exchange argument             |
| Greedy when DP needed   | Problem seems local-optimal       | Check if greedy fails on small examples  |
| Off-by-one in intervals | Inclusive vs exclusive endpoints  | Clarify: is [1,3] and [3,5] overlapping? |
| Forgetting to sort      | Rushing to implement              | Follow template: sort first              |
| Not handling ties       | Multiple items with same priority | Define consistent tiebreaker             |

---

## Common Misconceptions

### ❌ MISCONCEPTION: "If sorting helps, it's greedy"

✅ **REALITY**: Sorting is used in many paradigms (binary search, two pointers). Greedy specifically means making irrevocable locally-optimal choices.

**INTERVIEW TIP**: Explain why local choice leads to global optimum, not just that you're sorting.

### ❌ MISCONCEPTION: "Greedy always works for optimization problems"

✅ **REALITY**: Greedy only works when greedy choice property holds. Counterexample: 0/1 Knapsack, Coin Change with arbitrary denominations.

**INTERVIEW TIP**: Know counterexamples. If asked "why not greedy?", provide one.

### ❌ MISCONCEPTION: "If greedy gives a valid solution, it's optimal"

✅ **REALITY**: Greedy might give A solution but not THE optimal solution.

**Example**: Coin change with coins [1, 3, 4], amount = 6
- Greedy: 4 + 1 + 1 = 3 coins
- Optimal: 3 + 3 = 2 coins

### ❌ MISCONCEPTION: "Proving greedy correctness is unnecessary"

✅ **REALITY**: Without proof, you might be wrong. Interviewers may ask for justification.

**INTERVIEW TIP**: Have exchange argument ready: "If we don't make the greedy choice, we can swap and do no worse."

---

## Interview Tips for This Pattern

### How to Communicate Your Approach

1. **Identify pattern**: "This looks like a greedy problem because we're selecting activities/items with a clear ordering criterion."

2. **State the greedy strategy**: "My approach is to sort by [criterion] and greedily select the [first/best] that satisfies [constraint]."

3. **Justify correctness**: "This works because [exchange argument]: if we didn't pick the greedy choice, we could swap it in without making things worse."

4. **Analyze complexity**: "Sorting takes O(n log n), selection is O(n), so total is O(n log n)."

5. **Handle edge cases**: "I'll handle empty input and single element cases first."

### Common Follow-Up Questions

| Follow-Up                      | What They're Testing | How to Respond                     |
| ------------------------------ | -------------------- | ---------------------------------- |
| "Prove this is optimal"        | Formal reasoning     | Use exchange argument              |
| "What if greedy doesn't work?" | DP knowledge         | Explain how to use DP instead      |
| "Can you improve the time?"    | Optimization skills  | Consider if sorting can be avoided |
| "What if ties exist?"          | Attention to detail  | Define explicit tiebreaker         |
| "What about this input?"       | Edge case handling   | Walk through the algorithm         |

### Red Flags to Avoid

- **Assuming greedy works without justification**: Always briefly explain why
- **Using wrong sorting criterion**: Think about what "locally optimal" means
- **Forgetting to sort**: The sort step is crucial
- **Not recognizing when greedy fails**: Know classic counterexamples
- **Overcomplicating**: Greedy should be simple; if it's complex, reconsider

---

## Comparison with Related Patterns

| Aspect           | Greedy                       | Dynamic Programming          | Backtracking       |
| ---------------- | ---------------------------- | ---------------------------- | ------------------ |
| Decision style   | Irrevocable local choice     | Explore all, combine optimal | Try, recurse, undo |
| Time complexity  | Usually O(n log n)           | Usually O(n²) or O(n×W)      | Often O(2ⁿ)        |
| When to use      | Greedy choice property holds | Need to try all combinations | Need all solutions |
| Space complexity | Often O(1) or O(n)           | O(n) to O(n²)                | O(n) recursion     |
| Proof needed     | Yes (exchange argument)      | Recurrence correctness       | Completeness       |

### Decision Tree: Greedy vs DP vs Backtracking

```
                        Optimization Problem
                                │
                                ↓
                    ┌───────────────────────────┐
                    │ Does local optimal lead   │
                    │ to global optimal?        │
                    └───────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
         Yes (can prove)                  No / Unsure
                │                               │
                ↓                               ↓
        ┌───────────────┐           ┌─────────────────────┐
        │  Use GREEDY   │           │ Overlapping         │
        │     ✓         │           │ subproblems?        │
        └───────────────┘           └─────────────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                   Yes                     No
                                    │                       │
                                    ↓                       ↓
                            ┌───────────────┐   ┌─────────────────────┐
                            │   Use DP      │   │ Need all solutions? │
                            │     ✓         │   └─────────────────────┘
                            └───────────────┘               │
                                                ┌───────────┴────────┐
                                               Yes                  No
                                                │                    │
                                                ↓                    ↓
                                    ┌──────────────────┐  ┌──────────────────┐
                                    │ Use BACKTRACKING │  │ Use Divide &     │
                                    │        ✓         │  │ Conquer          │
                                    └──────────────────┘  └──────────────────┘
```

### When Greedy Fails → Use DP

| Problem                        | Greedy Fails Because                                   | DP Solution                                      |
| ------------------------------ | ------------------------------------------------------ | ------------------------------------------------ |
| 0/1 Knapsack                   | Can't take fractions                                   | dp[i][w] = max value with items 0..i, capacity w |
| Coin Change                    | Arbitrary denominations                                | dp[x] = min coins for amount x                   |
| Longest Increasing Subsequence | Local extension not globally optimal                   | dp[i] = LIS ending at i                          |
| Edit Distance                  | Local match/mismatch decisions don't accumulate simply | dp[i][j] = distance for prefixes                 |

---

## Real-World Applications

### Application 1: Task Scheduling (Operating Systems)

**Context**: CPU scheduling, job queues
**How Greedy Applies**: Shortest Job First (SJF), Priority Scheduling
**Why It Matters**: Minimizes average wait time, maximizes throughput

### Application 2: Network Routing (Dijkstra's Algorithm)

**Context**: Finding shortest paths in networks
**How Greedy Applies**: Always expand the closest unvisited node
**Why It Matters**: Powers GPS navigation, network packet routing

### Application 3: Data Compression (Huffman Coding)

**Context**: ZIP files, JPEG compression
**How Greedy Applies**: Build optimal prefix codes for characters
**Why It Matters**: Reduces file sizes by 50-90%

### Application 4: Resource Allocation (Load Balancing)

**Context**: Distributing tasks across servers
**How Greedy Applies**: Assign task to least-loaded server
**Why It Matters**: Maximizes throughput, minimizes latency

### Application 5: Minimum Spanning Tree (Network Design)

**Context**: Laying cables, designing networks
**How Greedy Applies**: Kruskal's/Prim's algorithms
**Why It Matters**: Minimizes infrastructure costs

---

## Practice Problems (Ordered by Difficulty)

### Warm-Up (Easy)
| #   | Problem                            | Key Concept         | LeetCode # |
| --- | ---------------------------------- | ------------------- | ---------- |
| 1   | Assign Cookies                     | Two-pointer greedy  | 455        |
| 2   | Lemonade Change                    | Simulation greedy   | 860        |
| 3   | Best Time to Buy and Sell Stock II | Collect all profits | 122        |
| 4   | Maximum Units on a Truck           | Sort by units       | 1710       |

### Core Practice (Medium)
| #   | Problem                          | Key Concept                | LeetCode # |
| --- | -------------------------------- | -------------------------- | ---------- |
| 1   | Jump Game                        | Reachability               | 55         |
| 2   | Jump Game II                     | BFS-like greedy            | 45         |
| 3   | Gas Station                      | Circular greedy            | 134        |
| 4   | Non-overlapping Intervals        | Activity selection variant | 435        |
| 5   | Partition Labels                 | Last occurrence tracking   | 763        |
| 6   | Task Scheduler                   | Frequency-based            | 621        |
| 7   | Queue Reconstruction by Height   | Sort + insert              | 406        |
| 8   | Minimum Arrows to Burst Balloons | Interval grouping          | 452        |

### Challenge (Hard)
| #   | Problem                        | Key Concept           | LeetCode # |
| --- | ------------------------------ | --------------------- | ---------- |
| 1   | Candy                          | Two-pass greedy       | 135        |
| 2   | Create Maximum Number          | Greedy + merge        | 321        |
| 3   | IPO                            | Heap-based greedy     | 502        |
| 4   | Minimum Cost to Hire K Workers | Ratio-based selection | 857        |

### Recommended Practice Order

1. **Assign Cookies (455)** - Basic two-pointer greedy
2. **Jump Game (55)** - Greedy reachability
3. **Non-overlapping Intervals (435)** - Classic activity selection
4. **Partition Labels (763)** - Creative greedy criterion
5. **Jump Game II (45)** - BFS-like minimum jumps
6. **Gas Station (134)** - Circular array greedy
7. **Task Scheduler (621)** - Frequency analysis
8. **Candy (135)** - Two-pass technique

---

## Code Templates

### Template 1: Interval Selection

```python
def interval_greedy(intervals: list[list[int]]) -> int:
    """
    Template for interval selection problems.

    Customize:
    - Sorting criterion (end time for max non-overlapping)
    - Selection condition
    - What to track/return
    """
    if not intervals:
        return 0

    # Sort by end time (for max non-overlapping)
    intervals.sort(key=lambda x: x[1])

    count = 1
    last_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start >= last_end:  # Non-overlapping
            count += 1
            last_end = end

    return count
```

### Template 2: Greedy with Heap

```python
import heapq

def heap_greedy(items: list, key_func) -> list:
    """
    Template for greedy problems requiring priority queue.

    Customize:
    - Heap ordering (min or max)
    - Selection/combination logic
    """
    # Create heap (negate for max-heap in Python)
    heap = [(key_func(item), item) for item in items]
    heapq.heapify(heap)

    result = []

    while len(heap) > 1:
        # Extract two smallest/largest
        key1, item1 = heapq.heappop(heap)
        key2, item2 = heapq.heappop(heap)

        # Combine (customize this)
        combined = combine(item1, item2)

        # Push back if needed
        heapq.heappush(heap, (key_func(combined), combined))

    return result
```

### Template 3: Two-Pass Greedy

```python
def two_pass_greedy(nums: list) -> list:
    """
    Template for problems requiring left-to-right then right-to-left passes.
    Example: Candy distribution, trapping rain water
    """
    n = len(nums)
    result = [1] * n  # or appropriate initial value

    # Left to right pass
    for i in range(1, n):
        if condition_left(nums, i):
            result[i] = update_left(result, i)

    # Right to left pass
    for i in range(n - 2, -1, -1):
        if condition_right(nums, i):
            result[i] = update_right(result, i)

    return result
```

### Template 4: Greedy with Sorting by Custom Key

```python
def custom_sort_greedy(items: list) -> any:
    """
    Template for problems with non-obvious sorting criterion.

    Key insight: Define what "locally optimal" means mathematically.
    """
    # Custom comparator: when should a come before b?
    # Example: For fractional knapsack, sort by value/weight ratio
    def sort_key(item):
        return item.value / item.weight  # Customize

    sorted_items = sorted(items, key=sort_key, reverse=True)

    result = initial_value
    for item in sorted_items:
        if can_select(item):
            result = update_result(result, item)

    return result
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GREEDY ALGORITHMS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ WHEN TO USE:                                                                │
│ • "Find minimum/maximum number of..."                                       │
│ • "Schedule/select to maximize..."                                          │
│ • Sorting reveals optimal order                                             │
│ • Local choice doesn't affect future options' optimality                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ KEY IDEA: Sort by criterion, select best available, never look back         │
├─────────────────────────────────────────────────────────────────────────────┤
│ COMPLEXITY: Usually O(n log n) for sort | O(n) for selection                │
├─────────────────────────────────────────────────────────────────────────────┤
│ THE 4-STEP TEMPLATE:                                                        │
│   1. IDENTIFY greedy criterion (what makes a choice "best"?)                │
│   2. SORT candidates by that criterion                                      │
│   3. ITERATE and greedily SELECT valid options                              │
│   4. PROVE correctness with exchange argument                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ COMMON CRITERIA:                                                            │
│   • Intervals: Sort by END time (activity selection)                        │
│   • Fractional: Sort by VALUE/WEIGHT ratio (knapsack)                       │
│   • Scheduling: Sort by DEADLINE or DURATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROOF TECHNIQUE - Exchange Argument:                                        │
│   1. Assume optimal solution O differs from greedy G                        │
│   2. Find first difference                                                  │
│   3. Swap O's choice with G's choice                                        │
│   4. Show result is no worse                                                │
│   5. Conclude: Greedy choice is safe                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ WATCH OUT FOR:                                                              │
│ • Wrong sorting criterion (prove it!)                                       │
│ • Problems where greedy fails (0/1 knapsack, coin change)                   │
│ • Forgetting to sort                                                        │
│ • Off-by-one in interval comparisons                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Connections to Other Patterns

### Prerequisites
- **Sorting** - Most greedy algorithms require sorting first
- **Heaps/Priority Queues** - For problems needing dynamic "best" selection
- **Basic proof techniques** - Exchange argument, induction

### Builds Upon
- Understanding of optimization problems
- Intuition for "locally optimal" choices
- Complexity analysis

### Leads To
- **Graph Algorithms**: Dijkstra, Prim, Kruskal use greedy
- **Approximation Algorithms**: Greedy often gives good approximations
- **Online Algorithms**: Make decisions without future knowledge

### Often Combined With
- **Sorting + Greedy**: Almost always
- **Heap + Greedy**: When "best" changes dynamically
- **Two Pointers + Greedy**: Matching problems
- **Binary Search + Greedy**: Greedy check in binary search predicate

---

## Summary: Key Takeaways

1. **Core Principle**: Make the locally optimal choice at each step, trusting it leads to global optimum.

2. **When to Use**: Problems where sorting reveals optimal order, and selecting "best" now doesn't hurt future options.

3. **Key Insight**: Greedy choice property—if you can swap any solution to include greedy choice without worsening, greedy works.

4. **Common Mistake**: Assuming greedy works without proof. Always verify with exchange argument or counterexample.

5. **Interview Tip**: State your greedy criterion explicitly ("sort by end time"), justify why it's optimal ("leaves maximum room for future"), then implement cleanly.

---

## Additional Resources

### Video Explanations
- **Abdul Bari** - Excellent visual explanations of classic greedy algorithms
- **MIT OpenCourseWare 6.046** - Rigorous proofs of greedy correctness
- **Back to Back SWE** - Interview-focused greedy problems

### Reading
- **Introduction to Algorithms (CLRS)** - Chapter 16, formal greedy framework
- **Algorithm Design (Kleinberg & Tardos)** - Great exchange argument examples
- **Competitive Programmer's Handbook** - Practical greedy techniques

### Interactive Practice
- **LeetCode** - Filter by "Greedy" tag
- **Codeforces** - Search "greedy" problems by rating
- **AtCoder** - ABC contests often have greedy problems
