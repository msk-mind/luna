import socket
import sys

"""
"Messages" are sent to this service of the format WRITE,<path_to_dir_or_file>,<id_to_attach>,<data_type>,<tag>

Example: python3 -m data_processing.services.upload \
localhost \
WRITE,/home/aukermaa/notes_about_scan.txt,2.25.323991087744237694972453104346210882376,ANNOTATION,test.text.andy
"""
host = sys.argv[1]
message = sys.argv[2]

io_service = (host, 5090)

client_socket = socket.socket()  # instantiate
client_socket.setblocking(1)
client_socket.connect(io_service)  # connect to the server
client_socket.send(message.encode())  # send message
client_socket.close()  # close the connection

