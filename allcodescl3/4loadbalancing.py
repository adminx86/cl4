def create_server(name, weight):
    return {"name": name, "weight": weight}

def create_load_balancer(servers):
    weighted_servers = []
    for server in servers:
        weighted_servers.extend([server] * server["weight"])
    return {"servers": weighted_servers, "current_index": 0}

def add_server(load_balancer, server):
    load_balancer["servers"].extend([server] * server["weight"])

def get_next_server(load_balancer):
    next_server = load_balancer["servers"][load_balancer["current_index"]]
    load_balancer["current_index"] = (load_balancer["current_index"] + 1) % len(load_balancer["servers"])
    return next_server

def prompt_server_info(index):
    name = input(f"Enter the name of server {index}: ")
    weight = int(input(f"Enter the weight of server {index}: "))
    return create_server(name, weight)

def assign_load(load_balancer, i):
    next_server = get_next_server(load_balancer)
    print(f"Load {i} assigned to server: {next_server['name']}")

if __name__ == "__main__":
    servers = []
    num_servers = int(input("Enter the number of servers: "))
    for i in range(1, num_servers + 1):
        servers.append(prompt_server_info(i))
    lb = create_load_balancer(servers)
    
    num_loads = int(input("Enter the number of loads: "))
    print("\nLoad balancing result:")
    for i in range(1, num_loads + 1):
        assign_load(lb, i)
