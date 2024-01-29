using HTTP
using Sockets
using Base.Threads

# Define a struct to represent a node
struct Node
    id::Int
    ip::String
    type::String
    shared_nonce::Threads.Atomic{Int}  # Shared nonce for redundancy
end

# Define key-value pair object
struct kvp
    key::String
    value::String
end

# Define the decentralized manager
struct DecentralizedManager
    nodes::Vector{Node}
    user_to_node::Vector{kvp}  # Dict to track user IDs to nodes (IPs)
end

# TODO: create manager
# Function to create a new node
function create_node(userid::String, manager::DecentralizedManager, ip::String, type::String)
    id = length(manager.nodes) + 1
    shared_nonce = Threads.Atomic{Int}(0)  # Initialize shared nonce
    node = Node(id, ip, type, shared_nonce)
    push!(manager.nodes, node)
    println("Node $id with IP $ip created.")
    return new_manager, node
end

# Function to remove a node
function remove_node(manager::DecentralizedManager, node::Node)
    filter!(n -> n !== node, manager.nodes)
    println("Node $node.id with IP $node.ip removed.")
end

# Function to add a user ID to node mapping
function add_user_to_node(manager::DecentralizedManager, user_id::String, node::Node)
    manager.user_to_node[user_id] = node.ip
    println("User $user_id is using Node $node.id with IP $node.ip")
end

# Function to simulate a user starting mining
function simulate_user_mining(manager::DecentralizedManager, user_id::String)
    node_ip = manager.user_to_node[user_id]
    println("User $user_id is using Node with IP $node_ip to mine.")
end

# Simulate manager tasks
function manage_nodes(manager::DecentralizedManager)
    while true
        sleep(1)  # Placeholder for manager tasks (e.g., checking node status)
        println("Manager task...")
    end
end
