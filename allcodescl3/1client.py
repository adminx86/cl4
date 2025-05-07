import xmlrpc.client

# Connect to the server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Get user input
n = int(input("Enter an integer to compute its factorial: "))

# Call remote method
result = proxy.factorial(n)

# Display result
print(f"Factorial of {n} is: {result}")

