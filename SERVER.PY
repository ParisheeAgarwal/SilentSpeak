import socket
import threading

HOST = '127.0.0.1'  
PORT = 12345

def handle_client(conn, addr):
    print('Connected by', addr)
    while True:
        data = conn.recv(1024)
        if not data:
            break
        
        print("Received data from client:", data.decode())

        for client_conn in client_connections:
            if client_conn != conn:  
                client_conn.sendall(data)

    print('Connection closed by', addr)
    conn.close()


client_connections = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    
    server_socket.listen()
    print('Server is listening on {}:{}'.format(HOST, PORT))
    
    while True:
        conn, addr = server_socket.accept()
        client_connections.append(conn)
        
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
