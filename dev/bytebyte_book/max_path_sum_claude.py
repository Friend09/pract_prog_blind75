class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_path_sum(root):
    """
    Find the maximum sum of a continuous path in a binary tree.
    A path can start and end at any node and must be continuous.

    Args:
        root: TreeNode - root of the binary tree

    Returns:
        int - maximum sum of any continuous path
    """
    if not root:
        return 0

    max_sum = float('-inf')

    def max_path_from_node(node):
        """
        Returns the maximum sum of a path starting from this node going down.
        Also updates the global max_sum considering paths through this node.
        """
        nonlocal max_sum

        if not node:
            return 0

        # Get maximum path sums from left and right subtrees
        # We take max with 0 to ignore negative paths
        left_max = max(0, max_path_from_node(node.left))
        right_max = max(0, max_path_from_node(node.right))

        # Maximum path sum through current node (connecting left and right)
        current_max = node.val + left_max + right_max

        # Update global maximum
        max_sum = max(max_sum, current_max)

        # Return maximum path sum starting from current node going down
        return node.val + max(left_max, right_max)

    max_path_from_node(root)
    return max_sum

def create_test_tree():
    """
    Creates the test tree from the example:
           5
          / \
       -10   8
       / \   / \
      1  -7 9   7
     /    \    / \
   11     10  6  -3
    """
    root = TreeNode(5)

    # Left subtree
    root.left = TreeNode(-10)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(-7)
    root.left.left.left = TreeNode(11)
    root.left.right.right = TreeNode(10)

    # Right subtree
    root.right = TreeNode(8)
    root.right.left = TreeNode(9)
    root.right.right = TreeNode(7)
    root.right.right.left = TreeNode(6)
    root.right.right.right = TreeNode(-3)

    return root

def print_tree_paths(root, path=[], all_paths=[]):
    """
    Helper function to visualize all possible paths in the tree
    """
    if not root:
        return

    path.append(root.val)

    if not root.left and not root.right:  # Leaf node
        all_paths.append(path.copy())
    else:
        if root.left:
            print_tree_paths(root.left, path, all_paths)
        if root.right:
            print_tree_paths(root.right, path, all_paths)

    path.pop()

# Test the function
if __name__ == "__main__":
    # Create test tree
    test_tree = create_test_tree()

    # Find maximum path sum
    result = max_path_sum(test_tree)

    print(f"Maximum path sum: {result}")
    print(f"Expected: 30")
    print(f"Test passed: {result == 30}")

    # Additional test cases
    print("\n--- Additional Test Cases ---")

    # Test case 1: Single node
    single_node = TreeNode(42)
    print(f"Single node (42): {max_path_sum(single_node)}")

    # Test case 2: All negative values
    negative_tree = TreeNode(-3)
    negative_tree.left = TreeNode(-1)
    negative_tree.right = TreeNode(-2)
    print(f"All negative (-3, -1, -2): {max_path_sum(negative_tree)}")

    # Test case 3: Simple positive path
    simple_tree = TreeNode(1)
    simple_tree.left = TreeNode(2)
    simple_tree.right = TreeNode(3)
    print(f"Simple tree (1, 2, 3): {max_path_sum(simple_tree)}")
