"""
Recursion Practice Examples
A collection of recursive functions to study and experiment with
"""


def section_header(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


# ============================================================================
# EXAMPLE 1: COUNTDOWN - The Simplest Recursion
# ============================================================================

def countdown(n):
    """
    Count down from n to 0, then print 'Blastoff!'
    
    Base case: n <= 0
    Recursive case: Print n, then call countdown(n-1)
    """
    if n <= 0:
        print("Blastoff! 🚀")
        return
    
    print(n)
    countdown(n - 1)


# ============================================================================
# EXAMPLE 2: FACTORIAL - Building Up a Result
# ============================================================================

def factorial(n):
    """
    Calculate n! = n × (n-1) × (n-2) × ... × 1
    
    Base case: n <= 1 returns 1
    Recursive case: n × factorial(n-1)
    """
    if n <= 1:
        return 1
    
    return n * factorial(n - 1)


# ============================================================================
# EXAMPLE 3: FIBONACCI - Multiple Recursive Calls
# ============================================================================

def fibonacci(n):
    """
    Calculate the nth Fibonacci number
    Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...
    
    Base cases: fib(0) = 0, fib(1) = 1
    Recursive case: fib(n-1) + fib(n-2)
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_memo(n, memo=None):
    """
    Optimized Fibonacci with memoization
    Much faster for large n!
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# ============================================================================
# EXAMPLE 4: LIST OPERATIONS
# ============================================================================

def sum_list(numbers):
    """
    Sum all numbers in a list recursively
    
    Base case: Empty list returns 0
    Recursive case: First element + sum of rest
    """
    if len(numbers) == 0:
        return 0
    
    return numbers[0] + sum_list(numbers[1:])


def list_length(lst):
    """Count elements in a list recursively"""
    if len(lst) == 0:
        return 0
    
    return 1 + list_length(lst[1:])


def find_maximum(lst):
    """
    Find the maximum value in a list recursively
    
    Base case: Single element list returns that element
    Recursive case: Max of first vs max of rest
    """
    if len(lst) == 1:
        return lst[0]
    
    max_of_rest = find_maximum(lst[1:])
    return lst[0] if lst[0] > max_of_rest else max_of_rest


def reverse_list(lst):
    """
    Reverse a list recursively
    
    Base case: Empty list returns empty list
    Recursive case: Last element + reverse of rest
    """
    if len(lst) == 0:
        return []
    
    return [lst[-1]] + reverse_list(lst[:-1])


# ============================================================================
# EXAMPLE 5: STRING OPERATIONS
# ============================================================================

def is_palindrome(s):
    """
    Check if a string is a palindrome (reads same forwards and backwards)
    
    Base case: String of length 0 or 1 is palindrome
    Recursive case: First == Last AND middle is palindrome
    """
    # Clean the string
    s = s.replace(" ", "").lower()
    
    if len(s) <= 1:
        return True
    
    if s[0] != s[-1]:
        return False
    
    return is_palindrome(s[1:-1])


def reverse_string(s):
    """Reverse a string recursively"""
    if len(s) <= 1:
        return s
    
    return s[-1] + reverse_string(s[:-1])


def count_char(s, char):
    """Count occurrences of a character in string"""
    if len(s) == 0:
        return 0
    
    count = 1 if s[0] == char else 0
    return count + count_char(s[1:], char)


# ============================================================================
# EXAMPLE 6: MATHEMATICAL OPERATIONS
# ============================================================================

def power(base, exponent):
    """
    Calculate base^exponent recursively
    Optimized: Uses half the recursion for even exponents
    
    Base case: base^0 = 1
    Recursive case: For even n: (base^(n/2))^2
                    For odd n: base × base^(n-1)
    """
    if exponent == 0:
        return 1
    
    # Optimization for even exponents
    if exponent % 2 == 0:
        half = power(base, exponent // 2)
        return half * half
    
    return base * power(base, exponent - 1)


def gcd(a, b):
    """
    Greatest Common Divisor using Euclidean algorithm
    
    Base case: gcd(a, 0) = a
    Recursive case: gcd(a, b) = gcd(b, a % b)
    """
    if b == 0:
        return a
    
    return gcd(b, a % b)


def sum_digits(n):
    """
    Sum all digits in a number
    Example: sum_digits(1234) = 10
    
    Base case: Single digit returns itself
    Recursive case: Last digit + sum of rest
    """
    if n < 10:
        return n
    
    return (n % 10) + sum_digits(n // 10)


# ============================================================================
# EXAMPLE 7: ADVANCED - BINARY SEARCH
# ============================================================================

def binary_search(arr, target, left=0, right=None):
    """
    Search for target in sorted array
    Returns index if found, -1 otherwise
    
    Base case: Search space empty (left > right)
    Recursive case: Search left or right half
    """
    if right is None:
        right = len(arr) - 1
    
    # Base case: not found
    if left > right:
        return -1
    
    # Find middle
    mid = (left + right) // 2
    
    # Found it!
    if arr[mid] == target:
        return mid
    
    # Search left half
    if arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    
    # Search right half
    return binary_search(arr, target, mid + 1, right)


# ============================================================================
# EXAMPLE 8: ADVANCED - PERMUTATIONS
# ============================================================================

def permutations(s):
    """
    Generate all permutations of a string
    
    Base case: Single char or empty has one permutation
    Recursive case: Pick each char, permute the rest
    """
    if len(s) <= 1:
        return [s]
    
    result = []
    for i in range(len(s)):
        current = s[i]
        remaining = s[:i] + s[i+1:]
        
        for perm in permutations(remaining):
            result.append(current + perm)
    
    return result


# ============================================================================
# EXAMPLE 9: DEBUGGING HELPER - TRACED FACTORIAL
# ============================================================================

def factorial_traced(n, depth=0):
    """
    Factorial with tracing to see the recursion in action
    Shows the call stack visually
    """
    indent = "  " * depth
    print(f"{indent}→ factorial({n})")
    
    if n <= 1:
        print(f"{indent}← returning 1")
        return 1
    
    result = n * factorial_traced(n - 1, depth + 1)
    print(f"{indent}← returning {result}")
    return result


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    """Run all examples"""
    
    # Test 1: Countdown
    section_header("1. COUNTDOWN")
    countdown(5)
    
    # Test 2: Factorial
    section_header("2. FACTORIAL")
    for n in [0, 1, 5, 10]:
        print(f"factorial({n}) = {factorial(n)}")
    
    # Test 3: Fibonacci
    section_header("3. FIBONACCI SEQUENCE")
    print("First 10 Fibonacci numbers:")
    for i in range(10):
        print(f"fib({i}) = {fibonacci(i)}", end="  ")
    print("\n")
    
    print("Compare speeds:")
    print(f"fibonacci(35) = {fibonacci(35)} (slow!)")
    print(f"fibonacci_memo(35) = {fibonacci_memo(35)} (fast!)")
    
    # Test 4: List Operations
    section_header("4. LIST OPERATIONS")
    test_list = [1, 2, 3, 4, 5]
    print(f"List: {test_list}")
    print(f"sum_list() = {sum_list(test_list)}")
    print(f"list_length() = {list_length(test_list)}")
    print(f"find_maximum() = {find_maximum([3, 7, 2, 9, 1])}")
    print(f"reverse_list() = {reverse_list(test_list)}")
    
    # Test 5: String Operations
    section_header("5. STRING OPERATIONS")
    test_words = ["racecar", "hello", "A man a plan a canal Panama"]
    for word in test_words:
        result = "✅ YES" if is_palindrome(word) else "❌ NO"
        print(f'is_palindrome("{word}") = {result}')
    
    print(f"\nreverse_string('hello') = {reverse_string('hello')}")
    print(f"count_char('mississippi', 's') = {count_char('mississippi', 's')}")
    
    # Test 6: Mathematical Operations
    section_header("6. MATHEMATICAL OPERATIONS")
    print(f"power(2, 10) = {power(2, 10)}")
    print(f"power(5, 3) = {power(5, 3)}")
    print(f"gcd(48, 18) = {gcd(48, 18)}")
    print(f"sum_digits(1234) = {sum_digits(1234)}")
    
    # Test 7: Binary Search
    section_header("7. BINARY SEARCH")
    sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    print(f"Array: {sorted_arr}")
    print(f"binary_search(arr, 7) = {binary_search(sorted_arr, 7)}")
    print(f"binary_search(arr, 10) = {binary_search(sorted_arr, 10)} (not found)")
    
    # Test 8: Permutations
    section_header("8. PERMUTATIONS")
    print(f"permutations('ABC') = {permutations('ABC')}")
    
    # Test 9: Traced Execution
    section_header("9. TRACED FACTORIAL (Watch the Call Stack!)")
    print("This shows how the recursion works step by step:\n")
    factorial_traced(4)
    
    section_header("✅ ALL TESTS COMPLETE!")
    print("\nNow try modifying the examples and creating your own!")
    print("Practice makes perfect! 💪")


if __name__ == "__main__":
    main()
