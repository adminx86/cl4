from xmlrpc.server import SimpleXMLRPCServer
import math

# Function to calculate factorial
def compute_factorial(n):
    if n < 0:
        return "Error: Factorial is not defined for negative numbers."
    return math.factorial(n)

# Create server
server = SimpleXMLRPCServer(("localhost", 8000))
print("RPC Server listening on port 8000...")

# Register function
server.register_function(compute_factorial, "factorial")

# Run the server
server.serve_forever()

