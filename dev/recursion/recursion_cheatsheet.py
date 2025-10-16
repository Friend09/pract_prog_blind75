"""
RECURSION QUICK REFERENCE CHEAT SHEET
=====================================

Use this as a quick reference when writing recursive functions!
"""

# ============================================================================
# THE RECURSION TEMPLATE
# ============================================================================

def recursive_function(input_data):
    """
    Every recursive function follows this pattern:
    
    1. BASE CASE(S) - Stop condition
    2. RECURSIVE CASE - Call itself with simpler input
    3. ENSURE PROGRESS - Move toward base case
    """
    
    # BASE CASE: When to stop
    if base_condition:
        return base_value
    
    # RECURSIVE CASE: Break down the problem
    result = process(input_data)
    
    # Call yourself with simpler input
    return combine(result, recursive_function(simpler_input))


# ============================================================================
# PATTERN 1: LINEAR RECURSION (Process one, recurse on rest)
# ============================================================================

def sum_list(lst):
    """Add all numbers in a list"""
    if not lst:                      # Base: Empty list
        return 0
    return lst[0] + sum_list(lst[1:])  # First + rest


def count_items(lst):
    """Count items in a list"""
    if not lst:
        return 0
    return 1 + count_items(lst[1:])


def list_contains(lst, target):
    """Check if list contains target"""
    if not lst:
        return False
    if lst[0] == target:
        return True
    return list_contains(lst[1:], target)


# ============================================================================
# PATTERN 2: DIVIDE AND CONQUER (Split in half)
# ============================================================================

def binary_search(arr, target, left, right):
    """Search sorted array by splitting in half"""
    if left > right:                 # Base: Empty range
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)


# ============================================================================
# PATTERN 3: MULTIPLE RECURSION (Tree-like)
# ============================================================================

def fibonacci(n):
    """Classic example with TWO recursive calls"""
    if n <= 1:                       # Base: First two numbers
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Two branches!


def tree_sum(node):
    """Sum all values in a binary tree"""
    if node is None:                 # Base: Empty tree
        return 0
    return node.value + tree_sum(node.left) + tree_sum(node.right)


# ============================================================================
# PATTERN 4: TAIL RECURSION (Last operation is recursive call)
# ============================================================================

def factorial_tail(n, accumulator=1):
    """Tail-recursive factorial (can be optimized)"""
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # Recursive call is LAST


def sum_tail(lst, accumulator=0):
    """Tail-recursive sum"""
    if not lst:
        return accumulator
    return sum_tail(lst[1:], accumulator + lst[0])


# ============================================================================
# PATTERN 5: BACKTRACKING (Try, recurse, undo if needed)
# ============================================================================

def solve_maze(maze, position, path):
    """
    Backtracking pattern for maze solving
    Try a move → Recurse → If fails, undo and try next
    """
    if is_goal(position):            # Base: Found solution
        return path
    
    for move in get_possible_moves(position):
        # Try this move
        new_pos = make_move(position, move)
        path.append(new_pos)
        
        result = solve_maze(maze, new_pos, path)
        if result:
            return result
        
        # Backtrack: undo the move
        path.pop()
    
    return None  # No solution found


# ============================================================================
# COMMON MISTAKES TO AVOID
# ============================================================================

# ❌ WRONG: No base case
def infinite_recursion(n):
    return n + infinite_recursion(n - 1)  # Never stops!


# ❌ WRONG: Not moving toward base case
def stuck_recursion(n):
    if n == 0:
        return 0
    return n + stuck_recursion(n)  # n never changes!


# ❌ WRONG: Forgetting to return
def missing_return(n):
    if n <= 1:
        return 1
    n * missing_return(n - 1)  # Missing 'return'!


# ✅ CORRECT: Has all three elements
def correct_recursion(n):
    if n <= 1:                    # 1. Base case
        return 1
    return n * correct_recursion(n - 1)  # 2. Recursive case + 3. Progress


# ============================================================================
# OPTIMIZATION: MEMOIZATION
# ============================================================================

def with_memoization(n, memo=None):
    """
    Cache results to avoid recalculating
    Transforms O(2^n) to O(n) for Fibonacci!
    """
    if memo is None:
        memo = {}
    
    # Check cache first
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Calculate and store
    memo[n] = with_memoization(n-1, memo) + with_memoization(n-2, memo)
    return memo[n]


# ============================================================================
# RECURSION TO ITERATION CONVERSION
# ============================================================================

# Recursive version
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)


# Equivalent iterative version
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# ============================================================================
# DEBUGGING TECHNIQUE: Add Tracing
# ============================================================================

def factorial_debug(n, depth=0):
    """Add depth parameter to visualize recursion"""
    indent = "  " * depth
    print(f"{indent}→ factorial({n})")
    
    if n <= 1:
        print(f"{indent}← return 1")
        return 1
    
    result = n * factorial_debug(n - 1, depth + 1)
    print(f"{indent}← return {result}")
    return result


# ============================================================================
# WHEN TO USE RECURSION VS ITERATION
# ============================================================================

"""
USE RECURSION WHEN:
✅ Working with trees or graphs
✅ Problem naturally divides into subproblems
✅ Backtracking is needed
✅ Code clarity is more important than performance
✅ Example: Tree traversal, DFS, permutations

USE ITERATION WHEN:
✅ Simple counting or looping
✅ Performance is critical
✅ Working with large datasets (stack overflow risk)
✅ Tail recursion that can be optimized
✅ Example: Summing array, factorial, Fibonacci (with memo)
"""


# ============================================================================
# COMPLEXITY QUICK REFERENCE
# ============================================================================

"""
COMMON RECURSION COMPLEXITIES:

1. Linear Recursion (one recursive call):
   Time: O(n), Space: O(n)
   Example: factorial, sum_list

2. Binary Recursion (two recursive calls):
   Time: O(2^n), Space: O(n)
   Example: naive fibonacci

3. Divide and Conquer (halving input):
   Time: O(log n), Space: O(log n)
   Example: binary_search

4. With Memoization:
   Time: O(n), Space: O(n)
   Example: fibonacci with memo

5. Tree Traversal:
   Time: O(n), Space: O(h) where h = height
   Example: tree operations
"""


# ============================================================================
# PRACTICE CHECKLIST
# ============================================================================

"""
Before writing a recursive function, ask:

1. ✅ What is my BASE CASE?
   - What's the simplest input I can handle directly?
   
2. ✅ What is my RECURSIVE CASE?
   - How do I break the problem into smaller pieces?
   
3. ✅ Am I making PROGRESS?
   - Does each call move toward the base case?
   
4. ✅ Do I need MEMOIZATION?
   - Am I recalculating the same values?
   
5. ✅ Is RECURSION the best approach?
   - Would iteration be clearer or more efficient?
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("This is a reference file - check out recursion_practice.py")
    print("for runnable examples!")
    print("="*60)
