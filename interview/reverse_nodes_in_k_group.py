class ListNode:
    def __init__(self, val, next_t=None):
        if isinstance(val, int):
            self.val = val
            self.next = next_t
        elif isinstance(val, list):
            self.val = val[0]
            self.next = None
            head = self
            for i in range(1, len(val)):
                node = ListNode(val[i])
                head.next = node
                head = head.next


class Solution:
    @staticmethod
    def reverse_list_node(head: ListNode, tail: ListNode):
        """反转子链表
        :param head:链表的头节点
        :param tail:链表的尾结点
        :return:反转之后链表的头结点和尾结点
        """
        prev = tail.next
        current = head
        while prev != tail:
            next_t = current.next
            current.next = prev
            prev = current
            current = next_t
        return tail, head

    def reverse_k_group(self, head: ListNode, k: int) -> ListNode:
        # 创建一个哑结点
        dummy_head = ListNode(0)
        dummy_head.next = head
        prev = dummy_head

        while head:
            tail = prev
            for i in range(k):
                tail = tail.next
                if not tail:
                    return dummy_head.next
            next_t = tail.next
            # 反转子链表
            head, tail = self.reverse_list_node(head, tail)
            # 更新节点的指向
            prev.next = head
            tail.next = next_t
            # 处理后面的子链表
            prev = tail
            head = tail.next

        return dummy_head.next


def main():
    l = [1, 2, 3, 4, 5]
    list_node = ListNode(l)
    obj = Solution()
    reverse_node = obj.reverse_k_group(list_node, 2)
    while reverse_node:
        print(reverse_node.val, end=',')
        reverse_node = reverse_node.next


if __name__ == '__main__':
    main()
