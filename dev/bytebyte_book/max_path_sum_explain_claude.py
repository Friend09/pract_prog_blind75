class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_path_sum_with_trace(root):
    """
    Maximum Path Sum with detailed execution trace
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    call_count = 0
    
    def max_path_from_node(node, depth=0):
        nonlocal max_sum, call_count
        call_count += 1
        
        indent = "  " * depth
        print(f"{indent}📞 Call #{call_count}: Processing node {node.val}")
        
        if not node:
            print(f"{indent}❌ Node is None, returning 0")
            return 0
        
        # STEP 1: Get maximum path sums from children
        print(f"{indent}🔍 Step 1: Exploring left child...")
        left_max = max(0, max_path_from_node(node.left, depth + 1) if node.left else 0)
        
        print(f"{indent}🔍 Step 1: Exploring right child...")
        right_max = max(0, max_path_from_node(node.right, depth + 1) if node.right else 0)
        
        print(f"{indent}📊 Results for node {node.val}:")
        print(f"{indent}   - Left max path: {left_max}")
        print(f"{indent}   - Right max path: {right_max}")
        
        # STEP 2: Calculate max path THROUGH this node
        current_max = node.val + left_max + right_max
        print(f"{indent}💡 Path through node {node.val}: {node.val} + {left_max} + {right_max} = {current_max}")
        
        # STEP 3: Update global maximum
        old_max = max_sum
        max_sum = max(max_sum, current_max)
        if max_sum != old_max:
            print(f"{indent}🎯 NEW GLOBAL MAX: {max_sum} (was {old_max})")
        else:
            print(f"{indent}⚡ Global max stays: {max_sum}")
        
        # STEP 4: Return max path FROM this node going down
        return_value = node.val + max(left_max, right_max)
        print(f"{indent}↩️  Returning: {node.val} + max({left_max}, {right_max}) = {return_value}")
        print(f"{indent}{'='*50}")
        
        return return_value
    
    print("🚀 Starting algorithm...\n")
    max_path_from_node(root)
    
    print(f"\n🏁 FINAL RESULT: {max_sum}")
    return max_sum

def create_simple_tree():
    """
    Simple tree for easier understanding:
        1
       / \
      2   3
    """
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    return root

def create_test_tree():
    """
    The original example tree
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

def explain_algorithm():
    """
    Conceptual explanation of the algorithm
    """
    print("🧠 ALGORITHM CONCEPT:")
    print("=" * 60)
    print("""
    The key insight: For each node, we need to consider TWO different things:
    
    1️⃣ PATHS THROUGH THIS NODE (for global maximum):
       - Connects left subtree → this node → right subtree
       - Formula: node.val + left_max + right_max
       - This could be our answer!
    
    2️⃣ PATHS FROM THIS NODE (to return to parent):
       - Goes down to either left OR right subtree
       - Formula: node.val + max(left_max, right_max)
       - This is what we return to the parent
    
    Why the difference?
    - A path can only be continuous (no splitting)
    - If we use both left AND right, we can't extend further up
    - If we use only one side, parent can extend the path
    """)
    
    print("\n🔄 RECURSION FLOW:")
    print("=" * 60)
    print("""
    1. Start at root
    2. For each node:
       a) Recursively solve left subtree
       b) Recursively solve right subtree  
       c) Calculate path THROUGH current node
       d) Update global maximum if needed
       e) Return path FROM current node (for parent to use)
    3. Base case: null nodes return 0
    """)

def visual_example():
    """
    Visual walkthrough with simple tree
    """
    print("\n🎨 VISUAL EXAMPLE:")
    print("=" * 60)
    print("""
    Tree:     1
             / \\
            2   3
    
    Execution order (post-order):
    1. Process node 2 (leaf) → returns 2
    2. Process node 3 (leaf) → returns 3  
    3. Process node 1 (root):
       - left_max = 2, right_max = 3
       - path through 1 = 1 + 2 + 3 = 6 ← This is maximum!
       - return from 1 = 1 + max(2,3) = 4
    
    Answer: 6 (path: 2 → 1 → 3)
    """)

if __name__ == "__main__":
    explain_algorithm()
    visual_example()
    
    print("\n" + "="*60)
    print("📋 SIMPLE EXAMPLE TRACE:")
    print("="*60)
    simple_tree = create_simple_tree()
    result1 = max_path_sum_with_trace(simple_tree)
    
    print("\n\n" + "="*60)
    print("📋 COMPLEX EXAMPLE TRACE (first few calls):")
    print("="*60)
    print("Note: This will be long! Focus on the pattern...")
    test_tree = create_test_tree()
    
    # Let's trace just a portion to avoid overwhelming output
    print("Tracing the original complex tree...")
    result2 = max_path_sum_with_trace(test_tree)
