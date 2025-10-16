"""
RECURSION PRACTICE EXERCISES
=============================

Work through these exercises to master recursion!
Solutions are provided at the bottom - try solving them first!
"""


# ============================================================================
# BEGINNER EXERCISES
# ============================================================================

def exercise_1_countdown():
    """
    EXERCISE 1: Countdown
    Write a function that counts down from n to 1 and prints "Done!"
    
    Example:
    countdown(3) should print:
    3
    2
    1
    Done!
    """
    # Your code here
    pass


def exercise_2_count_up(n):
    """
    EXERCISE 2: Count Up
    Write a function that counts UP from 1 to n
    
    Example:
    count_up(5) should print: 1 2 3 4 5
    
    Hint: Think about the base case and when to print
    """
    # Your code here
    pass


def exercise_3_power(base, exp):
    """
    EXERCISE 3: Power Function
    Calculate base^exp without using **
    
    Example:
    power(2, 3) = 8
    power(5, 0) = 1
    
    Hint: base^exp = base × base^(exp-1)
    """
    # Your code here
    pass


def exercise_4_sum_range(n):
    """
    EXERCISE 4: Sum Range
    Calculate sum of numbers from 1 to n
    
    Example:
    sum_range(5) = 1 + 2 + 3 + 4 + 5 = 15
    sum_range(3) = 1 + 2 + 3 = 6
    """
    # Your code here
    pass


def exercise_5_array_sum(arr):
    """
    EXERCISE 5: Array Sum
    Sum all numbers in an array
    
    Example:
    array_sum([1, 2, 3, 4]) = 10
    array_sum([]) = 0
    
    Hint: First element + sum of rest
    """
    # Your code here
    pass


# ============================================================================
# INTERMEDIATE EXERCISES
# ============================================================================

def exercise_6_reverse_string(s):
    """
    EXERCISE 6: Reverse String
    Reverse a string recursively
    
    Example:
    reverse_string("hello") = "olleh"
    reverse_string("a") = "a"
    
    Hint: Last character + reverse of rest
    """
    # Your code here
    pass


def exercise_7_count_vowels(s):
    """
    EXERCISE 7: Count Vowels
    Count vowels in a string recursively
    
    Example:
    count_vowels("hello") = 2  (e, o)
    count_vowels("aeiou") = 5
    
    Hint: Check first char, then recurse on rest
    """
    # Your code here
    pass


def exercise_8_flatten_list(nested_list):
    """
    EXERCISE 8: Flatten Nested List
    Convert nested list to flat list
    
    Example:
    flatten_list([1, [2, 3], [4, [5, 6]]]) = [1, 2, 3, 4, 5, 6]
    flatten_list([1, 2, 3]) = [1, 2, 3]
    
    Hint: Check if first element is a list
    """
    # Your code here
    pass


def exercise_9_gcd(a, b):
    """
    EXERCISE 9: Greatest Common Divisor
    Find GCD using Euclidean algorithm
    
    Example:
    gcd(48, 18) = 6
    gcd(100, 50) = 50
    
    Hint: gcd(a, b) = gcd(b, a % b), gcd(a, 0) = a
    """
    # Your code here
    pass


def exercise_10_is_sorted(arr):
    """
    EXERCISE 10: Check if Array is Sorted
    Return True if array is sorted in ascending order
    
    Example:
    is_sorted([1, 2, 3, 4]) = True
    is_sorted([1, 3, 2, 4]) = False
    is_sorted([]) = True
    
    Hint: Check if first two are in order, then check rest
    """
    # Your code here
    pass


# ============================================================================
# ADVANCED EXERCISES
# ============================================================================

def exercise_11_binary_search(arr, target, left=0, right=None):
    """
    EXERCISE 11: Binary Search
    Search for target in sorted array
    
    Example:
    binary_search([1, 3, 5, 7, 9], 5) = 2
    binary_search([1, 3, 5, 7, 9], 4) = -1
    
    Hint: Compare middle element, search left or right half
    """
    # Your code here
    pass


def exercise_12_merge_sort(arr):
    """
    EXERCISE 12: Merge Sort
    Sort an array using merge sort algorithm
    
    Example:
    merge_sort([3, 1, 4, 1, 5, 9, 2, 6]) = [1, 1, 2, 3, 4, 5, 6, 9]
    
    Hint: Split in half, sort each half, merge them
    You'll need a helper function to merge two sorted arrays
    """
    # Your code here
    pass


def exercise_13_generate_parentheses(n):
    """
    EXERCISE 13: Generate Parentheses
    Generate all valid combinations of n pairs of parentheses
    
    Example:
    generate_parentheses(2) = ["(())", "()()"]
    generate_parentheses(3) = ["((()))", "(()())", "(())()", "()(())", "()()()"]
    
    Hint: Use backtracking - keep track of open and close counts
    """
    # Your code here
    pass


def exercise_14_permutations(arr):
    """
    EXERCISE 14: Array Permutations
    Generate all permutations of an array
    
    Example:
    permutations([1, 2, 3]) = [
        [1, 2, 3], [1, 3, 2], [2, 1, 3], 
        [2, 3, 1], [3, 1, 2], [3, 2, 1]
    ]
    
    Hint: Pick each element, permute the rest
    """
    # Your code here
    pass


def exercise_15_tower_of_hanoi(n, source, destination, auxiliary):
    """
    EXERCISE 15: Tower of Hanoi
    Print moves to solve Tower of Hanoi puzzle
    
    Example:
    tower_of_hanoi(2, 'A', 'C', 'B') should print:
    Move disk 1 from A to B
    Move disk 2 from A to C
    Move disk 1 from B to C
    
    Hint: Move n-1 to auxiliary, move largest to destination,
          move n-1 from auxiliary to destination
    """
    # Your code here
    pass


# ============================================================================
# TEST FUNCTION
# ============================================================================

def run_tests():
    """Run basic tests for the exercises"""
    print("Testing your solutions...\n")
    
    # Test your solutions here as you complete them
    # Example:
    # print("Exercise 3:", exercise_3_power(2, 3))  # Should be 8
    
    print("\nComplete the exercises above!")


# ============================================================================
# SOLUTIONS (Try solving first before looking!)
# ============================================================================

def solutions():
    """
    SOLUTIONS - Only look after trying yourself!
    """
    
    # Solution 1: Countdown
    def countdown_solution(n):
        if n <= 0:
            print("Done!")
            return
        print(n)
        countdown_solution(n - 1)
    
    
    # Solution 2: Count Up
    def count_up_solution(n):
        if n <= 0:
            return
        count_up_solution(n - 1)
        print(n, end=" ")
    
    
    # Solution 3: Power
    def power_solution(base, exp):
        if exp == 0:
            return 1
        return base * power_solution(base, exp - 1)
    
    
    # Solution 4: Sum Range
    def sum_range_solution(n):
        if n <= 0:
            return 0
        return n + sum_range_solution(n - 1)
    
    
    # Solution 5: Array Sum
    def array_sum_solution(arr):
        if len(arr) == 0:
            return 0
        return arr[0] + array_sum_solution(arr[1:])
    
    
    # Solution 6: Reverse String
    def reverse_string_solution(s):
        if len(s) <= 1:
            return s
        return s[-1] + reverse_string_solution(s[:-1])
    
    
    # Solution 7: Count Vowels
    def count_vowels_solution(s):
        if len(s) == 0:
            return 0
        vowels = "aeiouAEIOU"
        count = 1 if s[0] in vowels else 0
        return count + count_vowels_solution(s[1:])
    
    
    # Solution 8: Flatten List
    def flatten_list_solution(nested_list):
        if len(nested_list) == 0:
            return []
        
        first = nested_list[0]
        rest = nested_list[1:]
        
        if isinstance(first, list):
            return flatten_list_solution(first) + flatten_list_solution(rest)
        else:
            return [first] + flatten_list_solution(rest)
    
    
    # Solution 9: GCD
    def gcd_solution(a, b):
        if b == 0:
            return a
        return gcd_solution(b, a % b)
    
    
    # Solution 10: Is Sorted
    def is_sorted_solution(arr):
        if len(arr) <= 1:
            return True
        if arr[0] > arr[1]:
            return False
        return is_sorted_solution(arr[1:])
    
    
    # Solution 11: Binary Search
    def binary_search_solution(arr, target, left=0, right=None):
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            return binary_search_solution(arr, target, left, mid - 1)
        else:
            return binary_search_solution(arr, target, mid + 1, right)
    
    
    # Solution 12: Merge Sort
    def merge_sort_solution(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort_solution(arr[:mid])
        right = merge_sort_solution(arr[mid:])
        
        # Merge helper function
        def merge(left, right):
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
        
        return merge(left, right)
    
    
    # Solution 13: Generate Parentheses
    def generate_parentheses_solution(n):
        result = []
        
        def backtrack(current, open_count, close_count):
            if len(current) == 2 * n:
                result.append(current)
                return
            
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)
            
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)
        
        backtrack("", 0, 0)
        return result
    
    
    # Solution 14: Permutations
    def permutations_solution(arr):
        if len(arr) <= 1:
            return [arr]
        
        result = []
        for i in range(len(arr)):
            current = arr[i]
            remaining = arr[:i] + arr[i+1:]
            
            for perm in permutations_solution(remaining):
                result.append([current] + perm)
        
        return result
    
    
    # Solution 15: Tower of Hanoi
    def tower_of_hanoi_solution(n, source, destination, auxiliary):
        if n == 1:
            print(f"Move disk 1 from {source} to {destination}")
            return
        
        tower_of_hanoi_solution(n - 1, source, auxiliary, destination)
        print(f"Move disk {n} from {source} to {destination}")
        tower_of_hanoi_solution(n - 1, auxiliary, destination, source)
    
    
    print("="*60)
    print("SOLUTION DEMONSTRATIONS")
    print("="*60)
    
    print("\n1. Countdown from 5:")
    countdown_solution(5)
    
    print("\n2. Count up to 5:")
    count_up_solution(5)
    print()
    
    print("\n3. Power (2^10):", power_solution(2, 10))
    
    print("4. Sum range (1 to 10):", sum_range_solution(10))
    
    print("5. Array sum ([1,2,3,4,5]):", array_sum_solution([1, 2, 3, 4, 5]))
    
    print("6. Reverse 'hello':", reverse_string_solution("hello"))
    
    print("7. Count vowels in 'hello':", count_vowels_solution("hello"))
    
    print("8. Flatten [1,[2,3],[4,[5,6]]]:", 
          flatten_list_solution([1, [2, 3], [4, [5, 6]]]))
    
    print("9. GCD(48, 18):", gcd_solution(48, 18))
    
    print("10. Is [1,2,3,4] sorted?", is_sorted_solution([1, 2, 3, 4]))
    
    print("11. Binary search for 7 in [1,3,5,7,9]:", 
          binary_search_solution([1, 3, 5, 7, 9], 7))
    
    print("12. Merge sort [3,1,4,1,5,9,2,6]:", 
          merge_sort_solution([3, 1, 4, 1, 5, 9, 2, 6]))
    
    print("13. Generate parentheses (n=2):", 
          generate_parentheses_solution(2))
    
    print("14. Permutations of [1,2,3]:", 
          permutations_solution([1, 2, 3]))
    
    print("\n15. Tower of Hanoi (n=3):")
    tower_of_hanoi_solution(3, 'A', 'C', 'B')


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("Work through the exercises above!")
    print("When ready, uncomment 'solutions()' below to see answers.")
    print("="*60 + "\n")
    
    # Uncomment to see solutions after trying:
    # solutions()
