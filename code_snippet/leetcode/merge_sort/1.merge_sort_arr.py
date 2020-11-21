class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    @staticmethod
    def merge_two_list(l1: ListNode, l2: ListNode) -> ListNode:
        pre_head = ListNode(-1)
        pre = pre_head

        while l1 and l2:
            if l1.val <= l2.val:
                pre.next = l1
                l1 = l1.next
            else:
                pre.next = l2
                l2 = l2.next
            pre = pre.next

        pre.next = l1 if l1 else l2

        return pre_head.next
