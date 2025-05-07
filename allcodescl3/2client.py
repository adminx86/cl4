# CLIENT CELL

import Pyro4

# Paste the URI from the server output
uri = "PYRO:obj_0f7c0de419004721ad557cb1763e60c6@localhost:55273"  # Replace with your actual URI

proxy = Pyro4.Proxy(uri)

str1 = input("Enter first string: ")
str2 = input("Enter second string: ")

result = proxy.concatenate(str1, str2)
print("Concatenated result:", result)
