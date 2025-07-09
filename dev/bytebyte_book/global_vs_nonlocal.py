# Global variable (module level)
counter = 0
print(f"Counter INITIAL STATE: {counter}")

def demonstrate_scopes():
    print("\n=== SCOPE DEMONSTRATION ===\n")

    # Example 1: Using global
    def example_global():
        global counter  # Refers to the module-level variable
        counter += 1
        print(f"Global counter: {counter}")

    # Example 2: Using nonlocal
    def example_nonlocal():
        local_var = 10  # This is in the enclosing scope

        def inner_function():
            nonlocal local_var  # Refers to local_var in the enclosing function
            local_var += 5
            print(f"Nonlocal variable: {local_var}")

        print(f"Before inner function: {local_var}")
        inner_function()
        print(f"After inner function: {local_var}")

    # Example 3: What happens without nonlocal/global
    def example_without_keywords():
        outer_var = 100

        def inner_function():
            # This creates a NEW local variable, doesn't modify outer_var
            outer_var = 200  # This is a different variable!
            print(f"Inner outer_var: {outer_var}")

        print(f"Before inner: {outer_var}")
        inner_function()
        print(f"After inner: {outer_var}")  # Still 100!

    # Example 4: Reading vs Writing
    def example_reading_vs_writing():
        outer_var = 50

        def can_read():
            # Can READ from enclosing scope without nonlocal
            print(f"Reading outer_var: {outer_var}")

        def cannot_write():
            # This would cause an error if uncommented:
            outer_var = 60  # UnboundLocalError!
            print(f"Cannot_Write outer_var: {outer_var}")
            # pass

        def can_write_with_nonlocal():
            nonlocal outer_var
            outer_var = 75  # Now this works!
            print(f"Modified outer_var: {outer_var}")

        can_read()
        cannot_write()
        can_write_with_nonlocal()
        print(f"Final outer_var: {outer_var}")

    # Run examples
    print("1. Global example:")
    example_global()
    example_global()

    print("\n2. Nonlocal example:")
    example_nonlocal()

    print("\n3. Without keywords:")
    example_without_keywords()

    print("\n4. Reading vs Writing:")
    example_reading_vs_writing()

def max_path_sum_explained():
    """
    Explaining the specific use in our binary tree code
    """
    print("\n=== IN OUR BINARY TREE CODE ===")

    max_sum = float('-inf')  # This is in the enclosing scope

    def max_path_from_node(node):
        nonlocal max_sum  # Refers to max_sum from the outer function

        # Without nonlocal, this line would create a NEW local variable
        # With nonlocal, it modifies the max_sum from the outer function
        max_sum = max(max_sum, 999)  # Example modification

        print(f"Modified max_sum: {max_sum}")

    print(f"Initial max_sum: {max_sum}")
    max_path_from_node(None)
    print(f"Final max_sum: {max_sum}")

# Alternative approaches without nonlocal
def alternative_approaches():
    print("\n=== ALTERNATIVES TO NONLOCAL ===")

    # Approach 1: Return values instead
    def approach_with_return():
        def helper(current_max):
            # Return the new maximum instead of modifying
            return max(current_max, 999)

        max_sum = float('-inf')
        max_sum = helper(max_sum)
        print(f"Using return values: {max_sum}")

    # Approach 2: Use a mutable object
    def approach_with_mutable():
        max_sum = [float('-inf')]  # List is mutable

        def helper():
            # Can modify list contents without nonlocal
            max_sum[0] = max(max_sum[0], 999)

        helper()
        print(f"Using mutable object: {max_sum[0]}")

    # Approach 3: Class-based approach
    class MaxFinder:
        def __init__(self):
            self.max_sum = float('-inf')

        def update_max(self, value):
            self.max_sum = max(self.max_sum, value)

    finder = MaxFinder()
    finder.update_max(999)
    print(f"Using class: {finder.max_sum}")

    approach_with_return()
    approach_with_mutable()

if __name__ == "__main__":
    demonstrate_scopes()
    max_path_sum_explained()
    alternative_approaches()
