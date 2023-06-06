# -*- coding: utf-8 -*-
import socket
import time


def debug00():
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


def debug01():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_address = ("localhost", 9091)
    client_socket.bind(client_address)
    client_socket.settimeout(10)

    server_address = ("localhost", 9090)

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
        print("-" * 36)
        receive_data, server = client_socket.recvfrom(1024)
        print(f"from:{server},content:{receive_data.decode()}\n")


if __name__ == '__main__':
    debug01()
