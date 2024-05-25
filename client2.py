import socket

# Define the server's IP address and port
SERVER_HOST = '127.0.0.1'  # Loopback address
SERVER_PORT = 12345

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # Connect to the server
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    while True:
        # Send data to the server
        data_to_send = input("Enter data to send: ")
        client_socket.sendall(data_to_send.encode())

        # Receive the list from the server
        received_data = client_socket.recv(1024)
        if received_data:
            print('Received list:', ''.join(eval(received_data.decode())))
