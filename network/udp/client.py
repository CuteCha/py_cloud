# -*- coding: utf-8 -*-
import socket
import time

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PORT = 9090
server_address = ("localhost", PORT)

while True:
    print("=" * 36)
    start = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))
    msg = input("content:")
    client_socket.sendto(msg.encode(), server_address)
    now = time.time()
    run_time = now - start
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)))
    print(f"run_time: {run_time} seconds\n")
