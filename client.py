import socket
import json


def receive_single_json_object(server_ip, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_ip, server_port))
        # Attempt to parse the accumulated data as JSON
        while True:
            try:
                data = s.recv(4096)
                radar_data = json.loads(data.decode())
                # Process the JSON data here
                print(f"Received radar data: {radar_data}\r")
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", e)


if __name__ == "__main__":
    server_ip = 'localhost'  # Change this to your server's IP address
    server_port = 5555  # Change this to your server's port
    receive_single_json_object(server_ip, server_port)
