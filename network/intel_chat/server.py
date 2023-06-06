import socket

address = ('192.134.163.105', 10000)  # local IP
re_addr = ("192.78.26.29", 10000)  # target IP
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def demo00():
    s.bind(address)
    while True:
        data, addr = s.recvfrom(2048)
        if not data:
            break
        print(f"from:{addr}\ncontent:{data.decode()}")
        reply_data = input("reply:")
        s.sendto(reply_data.encode("utf-8"), re_addr)

    s.close()


if __name__ == '__main__':
    demo00()
