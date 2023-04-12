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


# 打印链表的实用功能
def print_list(msg, head):
    print(msg, end=': ')
    ptr = head
    while ptr:
        print(ptr.data, end=' —> ')
        ptr = ptr.next
    print('None')


class Node:
    def __init__(self, data=None, d_next=None):
        self.data = data
        self.next = d_next


def reverse(head, m, n):
    if m > n:
        return head

    prev = None
    curr = head

    i = 1
    while curr is not None and i < m:
        prev = curr
        curr = curr.next
        i = i + 1

    start = curr
    end = None

    while curr is not None and i <= n:
        t_next = curr.next
        curr.next = end
        end = curr
        curr = t_next
        i = i + 1

    if start:
        start.next = curr
        if prev is None:
            head = end
        else:
            prev.next = end

    return head


def reverse2(head: Node, m: int, n: int) -> Node:
    if not head or not head.next or m >= n:
        return head

    dummy = Node(0)
    dummy.next = head
    start = dummy
    for i in range(m - 1):
        start = start.next

    end = cur = start.next
    pre = None
    for i in range(n - m + 1):
        t_next = cur.next
        cur.next = pre
        pre = cur
        cur = t_next
    start.next = pre
    end.next = cur

    return dummy.next


def reverse3(head):
    if head is None or head.next is None:
        return head

    pre = None
    cur = head
    h = head
    while cur:
        h = cur
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
    return h


def reverse4(head):
    if head is None or head.next is None:
        return head

    pre = None
    cur = head
    h = head
    print("=" * 72)
    print_list("h", h)
    print_list("cur", cur)
    print_list("pre", pre)
    while cur:
        h = cur
        tmp = h.next
        h.next = pre
        pre = h
        cur = tmp
        print("=" * 72)
        print_list("h", h)
        print_list("cur", cur)
        print_list("pre", pre)
        print_list("tmp", tmp)

        # print(f"cur={cur.data if cur else cur}")
        # print(f"pre={pre.data}")
    return h


def reverse5(head):
    if head is None or head.next is None:
        return head

    pre = None
    cur = head

    while cur:
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
        print("=" * 72)
        print_list("cur", cur)
        print_list("pre", pre)
        print_list("tmp", tmp)

        # print(f"cur={cur.data if cur else cur}")
        # print(f"pre={pre.data}")
    return pre


def main2():
    head = None
    for i in reversed(range(7)):
        head = Node(i + 1, head)

    (m, n) = (2, 5)

    print_list('Original linked list', head)
    # head = reverse2(head, m, n)
    head = reverse5(head)
    # print_list('Reversed linked list', head)


if __name__ == '__main__':
    main2()
