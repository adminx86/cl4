#pip install Pyro4   -do this before running this code

# SERVER CELL


import Pyro4

@Pyro4.expose
class StringConcatenator:
    def concatenate(self, str1, str2):
        print(f"Received: {str1}, {str2}")
        return str1 + str2

def start_server():
    daemon = Pyro4.Daemon(host="localhost")  # Bind to localhost
    uri = daemon.register(StringConcatenator)
    print("Server running. URI:", uri)
    
    daemon.requestLoop()  # Keep the server running

start_server()

#Run the Server

#From the terminal output, copy the exact URI — it will look like this:

#Server running. URI: PYRO:obj_193c55e671d04d8eb67278354873f956@localhost:50111

#Update this line in your client.py

#run the client