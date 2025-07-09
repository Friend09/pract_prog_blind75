from typing import List, Optional

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_binary_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Build a binary tree from preorder and inorder traversal arrays.

    Args:
        preorder: List of values in preorder traversal
        inorder: List of values in inorder traversal

    Returns:
        Root node of the constructed binary tree
    """
    if not preorder or not inorder:
        return None

    # Create index map for O(1) lookups in inorder array
    inorder_map = {val: i for i, val in enumerate(inorder)}

    # Use list to make preorder_index mutable across recursive calls
    preorder_index = [0]

    def build_subtree(left: int, right: int) -> Optional[TreeNode]:
        """
        Recursively build subtree for the given inorder range.

        Args:
            left: Left boundary (inclusive) in inorder array
            right: Right boundary (inclusive) in inorder array

        Returns:
            Root node of the subtree
        """
        # Base case: empty range
        if left > right:
            return None

        # Current root is next element in preorder traversal
        root_val = preorder[preorder_index[0]]
        preorder_index[0] += 1

        # Find root position in inorder array
        root_inorder_index = inorder_map[root_val]

        # Create root node
        root = TreeNode(root_val)

        # Build left subtree (elements before root in inorder)
        root.left = build_subtree(left, root_inorder_index - 1)

        # Build right subtree (elements after root in inorder)
        root.right = build_subtree(root_inorder_index + 1, right)

        return root

    return build_subtree(0, len(inorder) - 1)


# Example usage and testing
def print_inorder(root: Optional[TreeNode]) -> List[int]:
    """Helper function to verify the tree construction"""
    result = []

    def inorder_traverse(node):
        if node:
            inorder_traverse(node.left)
            result.append(node.val)
            inorder_traverse(node.right)

    inorder_traverse(root)
    return result

def print_preorder(root: Optional[TreeNode]) -> List[int]:
    """Helper function to verify the tree construction"""
    result = []

    def preorder_traverse(node):
        if node:
            result.append(node.val)
            preorder_traverse(node.left)
            preorder_traverse(node.right)

    preorder_traverse(root)
    return result


# Test the implementation
if __name__ == "__main__":
    # Test case 1
    preorder1 = [3, 9, 20, 15, 7]
    inorder1 = [9, 3, 15, 20, 7]

    root1 = build_binary_tree(preorder1, inorder1)
    print("Test 1:")
    print(f"Original preorder: {preorder1}")
    print(f"Original inorder:  {inorder1}")
    print(f"Built preorder:    {print_preorder(root1)}")
    print(f"Built inorder:     {print_inorder(root1)}")
    print()

    # Test case 2
    preorder2 = [1, 2, 4, 5, 3, 6]
    inorder2 = [4, 2, 5, 1, 6, 3]

    root2 = build_binary_tree(preorder2, inorder2)
    print("Test 2:")
    print(f"Original preorder: {preorder2}")
    print(f"Original inorder:  {inorder2}")
    print(f"Built preorder:    {print_preorder(root2)}")
    print(f"Built inorder:     {print_inorder(root2)}")
