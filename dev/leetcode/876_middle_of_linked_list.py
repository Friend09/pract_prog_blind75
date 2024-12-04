"""
Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.
"""

# NOTE: the below is assuming the head as list but need to use linked list, so come back to this later

def middleNode(head):
    # get the lenght of head
    head_length = len(head)
    # check if length is odd
    if head_length % 2 == 1:
        idx = head_length // 2  # since we dont use math module
        return head[idx:]
    # else - even
    else:
        idx = int(head_length / 2)
        return head[idx:]


head = [1, 2, 3, 4, 5, 6]
print(middleNode(head))
