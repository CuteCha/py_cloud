class Node:
    def __init__(self, value: int):
        self.value = value
        self.next = None


def creat_link(arr: list[int], pos: int = -1) -> Node:
    if not arr:
        return None
    nodes = [Node(item) for item in arr]
    n = len(nodes)
    head = nodes[0]
    tmp = head
    for i in range(1, n):
        tmp.next = nodes[i]
        tmp = tmp.next

    if 0 < pos < n:
        tmp.next = nodes[pos]

    return head


def print_link(head: Node):
    tmp = head
    while tmp:
        print(f"{tmp.value}->", end="")
        tmp = tmp.next
    print("None")


def print_link_with_addr(head: Node):
    tmp = head
    while tmp:
        print(f"{tmp.value}({hex(id(tmp))})->", end="")
        tmp = tmp.next
    print("None")


def find_mid(head: Node) -> Node:
    if not head or not head.next:
        return head
    prev = None
    slow = head
    fast = head

    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next

    return prev


def merge(left: Node, right: Node) -> Node:
    l = left
    r = right
    d = Node(-1)
    c = d
    while l and r:
        if l.value < r.value:
            c.next = l
            l = l.next
        else:
            c.next = r
            r = r.next
        c = c.next

    if l:
        c.next = l
    if r:
        c.next = r

    return d.next


def merge_sort(head: Node) -> Node:
    if not head or not head.next:
        return head

    mid = find_mid(head)
    right = mid.next
    mid.next = None
    left = head

    l = merge_sort(left)
    r = merge_sort(right)

    return merge(l, r)


def _test_merge_sort():
    arr = [6, 1, 5, 2, 7, 3, 9, 8, 4, 6, 7]
    head = creat_link(arr)
    print_link(head)
    print_link(merge_sort(head))


def has_circle(head: Node) -> bool:
    if not head or not head.next:
        return False

    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False


def find_circle(head: Node) -> Node:
    if not head or not head.next:
        return None
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break

    p = head
    q = slow
    while p != q:
        p = p.next
        q = q.next

    return p


def _test_find_circle():
    arr = [6, 1, 5, 2, 7, 3, 9, 8, 4, 6, 7]
    head = creat_link(arr, 6)
    # print_link(head)
    print(has_circle(head))
    p = find_circle(head)
    print(p.value)


def reverse_link(head: Node) -> Node:
    if not head or not head.next:
        return head

    prev = None
    cur = head
    while cur:
        tmp = cur.next
        cur.next = prev
        prev = cur
        cur = tmp

    return prev


def link_add(a: Node, b: Node) -> Node:
    ar = reverse_link(a)
    br = reverse_link(b)

    return reverse_link(link_add_r(ar, br))


def link_add_r(a: Node, b: Node) -> Node:
    ta = a
    tb = b
    d = Node(-1)
    td = d
    c = 0

    while ta and tb:
        tmp = ta.value + tb.value + c
        c = tmp // 10
        s = tmp % 10
        td.next = Node(s)
        td = td.next
        ta = ta.next
        tb = tb.next

    ll = ta if ta else tb

    while ll:
        tmp = ll.value + c
        c = tmp // 10
        s = tmp % 10
        td.next = Node(s)
        td = td.next
        ll = ll.next

    if c > 0:
        td.next = Node(c)

    return d.next


def _test_link_add():
    a = [1, 9, 9]
    b = [9]
    ha = creat_link(a)
    hb = creat_link(b)
    print_link(ha)
    print_link(hb)

    print_link(link_add(ha, hb))


def _test_link_add_r():
    a = [1, 9, 9]
    b = [9]

    ha = creat_link(a)
    hb = creat_link(b)
    print_link(ha)
    print_link(hb)

    print_link(link_add_r(ha, hb))


def find_intersection_node(head1: Node, head2: Node) -> Node:
    if not head1 or not head2:
        return None

    t1 = head1
    t2 = head2
    while t1 != t2:
        t1 = t1.next if t1 else head2
        t2 = t2.next if t2 else head1

    return t1


def _test_find_intersection_node():
    arr1 = [1, 3, 5, 7, 9]
    arr2 = [0, 2, 4, 6, 8]
    arr3 = [11, 12, 13]

    head1 = creat_link(arr1)
    head2 = creat_link(arr2)
    head3 = creat_link(arr3)

    print_link(head1)
    print_link(head2)
    q = find_intersection_node(head1, head2)
    if q:
        print(q.value)
    else:
        print(None)

    l1 = head1
    t1 = l1
    while t1.next:
        t1 = t1.next
    t1.next = head3

    l2 = head2
    t2 = l2
    while t2.next:
        t2 = t2.next
    t2.next = head3

    print_link_with_addr(l1)
    print_link_with_addr(l2)
    p = find_intersection_node(l1, l2)
    if p:
        print(p.value)
    else:
        print(None)


if __name__ == '__main__':
    _test_find_intersection_node()
