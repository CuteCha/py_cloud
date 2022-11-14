# -*- coding: utf-8 -*-

import socket
import time

PORT = 9090
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("0.0.0.0", PORT)
server_socket.bind(address)
server_socket.settimeout(10)

while True:
    try:
        now = time.time()
        print("=" * 36)
        receive_data, client = server_socket.recvfrom(1024)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)))
        print(f"from:{client},content:{receive_data.decode()}\n")
    except socket.timeout:
        print("timeout\n")
