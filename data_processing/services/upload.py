import socket
import sys

"""
"Messages" are sent to this service of the format WRITE,<path_to_dir_or_file>,<id_to_attach>,<data_type>,<tag>

Example: python3 -m data_processing.services.upload \
WRITE,/home/aukermaa/notes_about_scan.txt,2.25.323991087744237694972453104346210882376,ANNOTATION,test.text.andy
"""
io_service   = ("pllimsksparky1", 5090)
message = sys.argv[1]  # take input
client_socket = socket.socket()  # instantiate
client_socket.setblocking(1)
client_socket.connect(io_service)  # connect to the server
client_socket.send(message.encode())  # send message
client_socket.close()  # close the connection

