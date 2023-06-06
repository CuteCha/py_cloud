import socket

address = ('192.134.163.105', 10000)  # target IP
re_addr = ('192.78.26.29', 10000)  # local IP
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def demo00():
    s.bind(re_addr)
    while True:
        data = input("input:")
        if not data:
            break
        s.sendto(data.encode("utf-8"), address)
        receive_data, addr = s.recvfrom(2048)
        if receive_data:
            print(f"from:{addr}\ncontent: {receive_data.decode()}")

    s.close()


if __name__ == '__main__':
    demo00()
