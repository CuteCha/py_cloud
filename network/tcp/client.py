# -*- coding: utf-8 -*-
import socket

HOST = '0.0.0.0'
PORT = 9090

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    out_data = input(f"message: ")
    print(f"send: {out_data}")
    s.send(out_data.encode())
    in_data = s.recv(1024)
    if len(in_data) == 0:
        s.close()
        print("server closed connection.")
        break
    print(f"recv: {in_data.decode()}")
