import socket
import struct
import threading
from time import sleep

# create a c++ struct like the message in the msclient source code
# the original struct is copied below for convenience
"""
typedef struct mscl_nachricht { char sender[50]; //String mit 50 Zeichen,
					 char empfaenger[50]; //String mit 50 Zeichen
					 WORD auftrnr; //eindeutige Nummer für aktuelles Segment (2 Byte)
					 BYTE dienstnr; //Kennung des Datenpakets
					 char daten[256]; //Zeichenkette mit 256 Zeichen
} t_mscl_nachricht;
"""
# word <=> 2 bytes
# byte <=> 1 byte (self-explanatory)

format = "50s 50s H B 256s"  # h is for a short int (2 bytes), b is for signed char (1 byte)
message_struct = struct.Struct(format)

def connect_moses_socket(host, port):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    return s

def send_moses_message(con_socket, client_name, server_name, service_number, task_number, data, ):
    
    # convert the input data into a bytecode message in the form of the above struct
    byte_message = message_struct.pack(client_name.encode(), server_name.encode(), task_number, service_number, data.encode())

    # send message
    con_socket.send(byte_message)

def receive_moses_message(con_socket):

    while True:
        # await response
        response = message_struct.unpack(con_socket.recv(4096))

        # some cleanup of the response to make it more human readable
        # the enconding for the decode() method may throw errors in some cases, latin-1 seems to work though
        response_sender = response[0].decode('latin-1').split("\x00",1)[0] 
        response_target = response[1].decode('latin-1').split("\x00",1)[0] 
        response_task_number = response[2]
        response_service_number = response[3]
        response_data = response[4].decode('latin-1').split("\x00",1)[0] 

        print("-------------------------")
        print("Message response:")
        print("Sender: " + response_sender)
        print("Target: " + response_target)
        print("Task number: " + str(response_task_number))
        print("Service number: " + str(response_service_number))
        print("Data: " + response_data)
        print("-------------------------")

if __name__ == "__main__":
    
    host = socket.gethostbyname("localhost")
    port = 1200

    con_socket = connect_moses_socket(host, port)

    listener = threading.Thread(target = receive_moses_message, args=(con_socket,))
    listener.daemon = True
    listener.start()

    while True:
        inp = input("Enter client name, server name, service number, task number, data separated only by a comma (no spaces):\n")
        inp = inp.split(",")
        
        client_name = inp[0]
        server_name = inp[1]
        service_number = int(inp[2])
        task_number = int(inp[3])
        data = inp[4]

        #client_name = "PierceCSL"
        #server_name = "MOSES Server"

        #service_number = 120
        #task_number    = 0

        #data = "C:/Data/Test/hallo_welt.py"

        print(listener.is_alive())
        send_moses_message(con_socket, client_name, server_name, service_number, task_number, data)
        sleep(0.5)





