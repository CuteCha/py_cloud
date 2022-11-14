# -*- coding: utf-8 -*-
import socket

HOST = '0.0.0.0'
PORT = 9090

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)

print(f"server start at: {HOST}:{PORT}")
print(f"wait for connection...")

while True:
    conn, addr = s.accept()
    print(f"connected by {addr}")

    while True:
        in_data = conn.recv(1024)
        if len(in_data) == 0:
            conn.close()
            print("client closed connection.")
            break
        print(f"recv: {in_data.decode()}")

        out_data = f"echo {in_data.decode()}"
        conn.send(out_data.encode())
