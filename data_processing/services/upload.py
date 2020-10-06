import socket
import sys

io_service   = ("pllimsksparky1:5090".split(":")[0], int("pllimsksparky1:5090".split(":")[1]))
message = sys.argv[1]  # take input
client_socket = socket.socket()  # instantiate
client_socket.setblocking(1)
client_socket.connect(io_service)  # connect to the server
client_socket.send(message.encode())  # send message
client_socket.close()  # close the connection

