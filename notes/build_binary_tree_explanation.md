# Building a Binary Tree: A Step-by-Step Guide 🌳

## What are we trying to do?

We have two lists of numbers that tell us about a tree:

- **Preorder**: The order we visit nodes starting from the root (parent first, then children)
- **Inorder**: The order we visit nodes going left-to-right across the tree

**Goal**: Use these two lists to rebuild the exact same tree!

## Example: Let's build this tree

```
    3
   / \
  9   20
     /  \
    15   7
```

**Given:**

- Preorder: [3, 9, 20, 15, 7] ← Root first, then left, then right
- Inorder: [9, 3, 15, 20, 7] ← Left, then root, then right

---

## Step 1: The Big Insight 💡

**Key Discovery:**

- The **first number in preorder** is always the root of the tree!
- In **inorder**, everything to the left of the root belongs to the left subtree
- Everything to the right of the root belongs to the right subtree

## Step 2: Let's trace through our example

### Round 1: Find the main root

- Preorder: [**3**, 9, 20, 15, 7] ← First item (3) is our root!
- Inorder: [9, **3**, 15, 20, 7] ← Find where 3 is located

```
Root = 3
Left subtree will contain: [9]        (everything left of 3 in inorder)
Right subtree will contain: [15,20,7] (everything right of 3 in inorder)
```

### Round 2: Build the left subtree

- Remaining preorder: [9, 20, 15, 7] ← Next item (9) is root of left subtree
- Left subtree inorder: [9] ← Only has 9, so it's a leaf node

```
    3
   /
  9      ← This becomes our left child
```

### Round 3: Build the right subtree

- Remaining preorder: [20, 15, 7] ← Next item (20) is root of right subtree
- Right subtree inorder: [15, 20, 7] ← Find where 20 is located

```
Right subtree root = 20
Left of 20: [15]  ← Will be 20's left child
Right of 20: [7]  ← Will be 20's right child
```

### Round 4 & 5: Fill in the remaining nodes

- Node 15: Next in preorder, goes to left of 20
- Node 7: Last in preorder, goes to right of 20

**Final tree:**

```
    3
   / \
  9   20
     /  \
    15   7
```

---

## Step 3: The Code Explained 🔍

### The Setup

```python
# Create a map to quickly find positions in inorder
inorder_map = {val: i for i, val in enumerate(inorder)}
# Example: {9: 0, 3: 1, 15: 2, 20: 3, 7: 4}

# Keep track of where we are in preorder
preorder_index = [0]  # Using a list so we can change it in recursion
```

### The Recursive Function

```python
def build_subtree(left: int, right: int):
```

- `left` and `right` define the boundaries in the inorder array
- We only work with elements between these positions

### Step-by-step process:

1. **Check if we're done**: If `left > right`, no more nodes to add
2. **Get the root**: Take the next item from preorder
3. **Find the split**: Look up where this root appears in inorder
4. **Create the node**: Make a new TreeNode with this value
5. **Build left side**: Recursively build everything to the left of root
6. **Build right side**: Recursively build everything to the right of root

### Example walkthrough:

```
Call 1: build_subtree(0, 4)  # Full range [9,3,15,20,7]
  → root = 3 (preorder[0])
  → 3 is at position 1 in inorder
  → Build left: build_subtree(0, 0)   # Just [9]
  → Build right: build_subtree(2, 4)  # [15,20,7]

Call 2: build_subtree(0, 0)  # Just [9]
  → root = 9 (preorder[1])
  → 9 is at position 0
  → No left child (0 > -1)
  → No right child (1 > 0)
  → Return node with value 9

Call 3: build_subtree(2, 4)  # [15,20,7]
  → root = 20 (preorder[2])
  → 20 is at position 3
  → Build left: build_subtree(2, 2)   # Just [15]
  → Build right: build_subtree(4, 4)  # Just [7]
```

---

## The Magic Trick ✨

The algorithm works because:

1. **Preorder gives us the order to create nodes** (parent before children)
2. **Inorder tells us which nodes go left vs right** of each parent
3. **We use this info to recursively split the problem** into smaller pieces

It's like having assembly instructions that tell you:

- Which piece to pick up next (preorder)
- Where to place it relative to what you've already built (inorder)

Pretty cool, right? 🚀
