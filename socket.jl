# Required imports
import Base: +, -

# Required files
include("compute.jl")
include("crystal.jl")
include("key.jl")
include("mongodb.jl")
# include("stark.jl")
include("zk-ecp.jl")
include("manager.jl")

# Required packages
using SHA
using Random
using Base64
using Nemo
using Elliptic
using JSON

using HTTP.Servers
using HTTP.Messages
using Sockets
using Base.Threads
using CUDA

# using Libsecp256k1

# using compute_mod
# using crystal_mod
# using key_mod
# using mongo_mod
# using stark_mod
# using zk_mod

# PROTOTYPE System Specs
# Get the operating system name
# os_name = Sys.KERNEL

# Get the endianness of the system (either :big or :little)
# endianness = Sys.endianness()

# Get information about CPU cores
# cpu_info = Sys.cpu_info()

# Get total and available system memory
# memory_info = SysCommands.free()

# Get information about available GPUs
gpu_info = CUDA.devices()
# for (i, device) in enumerate(gpu_info)
#     println("GPU $i:")
#     println("  Name: $(CUDA.name(device))")
#     println("  Compute Capability: $(CUDA.compute(device))")
#     println("  Total Memory: $(CUDA.memory(device) / 1024^2) MB")
# end

# Define the port for the server to listen on
const port = 9111
const ALLOWED_IPS = ["127.0.0.1"]  # Set of allowed IP addresses

# Define the prime p and the number of iterations
# const p = 32416187567
const iterations = 1000

# Define constant l angle 
const deg = 7

# Initialize a context
ctx = SHA3_512_CTX()
# const ctx = zeros(UInt8, 4)

# nonce
const global_nonce = Base.RefValue(0.001)

# Function to increment nonce
function increment_nonce()
    global_nonce[] += 1
    return floor.(global_nonce[])
end

# Implement chain image
artificial_chain = []

# Define the CSIDH parameters
# const A = 6
# const E = 1

# Define a prime number for the base field
# Define the prime field parameters
p = 32416187567

# Define the coefficients for complex multiplication
D = -121665 / 121666
I = -13.249
F = GF(p)
rho = GF(p)(-121665) / GF(p)(121666) # constant

# Prototype
# I, x1 = FiniteField(ZZ(p), -1, "x")
# F, x2 = FiniteField(ZZ(p), 1, "x")

# Compute the j-invariant for E (iso proof)
j_invariant_E1 = 1728 * (4*D^3) / (4*D^3 + 27*I^2)

# Calculate a and b
a = I
b = D / (D + 1)

# Define the secret key structure
struct SecretKey
    data::Vector{Int}
end

# Define the public key structure
struct PublicKey
    data::Vector{Int}
end

# Define a custom type for signatures
# struct Signature
#     data::Vector{String}
# end

# Define a block structure
struct XBlock
    nonce::Float64
    erp_codec::String
    previous_hash::String
    timestamp::DateTime
    consensus::Vector{String} # Change data to hold an array of signatures
    hash::String
    reward::Float64
end

# Define a struct to represent a node
struct Node
    id::Int
    ip::String
    type::String
    shared_nonce::Threads.Atomic{Int}  # Shared nonce for redundancy
end

struct Wallet
    private
    public
end

struct Point
    x::BigInt
    y::BigInt
end

struct MerkleNode
    hash::String
end

function build_merkle_tree(transactions)
    if length(transactions) == 1
        return [MerkleNode(sha256(transactions[1]))]
    end
    
    new_level = []
    for i in 1:2:length(transactions)-1
        left_hash = sha256(transactions[i])
        right_hash = sha256(transactions[i+1])
        push!(new_level, MerkleNode(sha256(left_hash * right_hash)))
    end
    
    return build_merkle_tree(new_level)
end

# Perform a Diffie-Hellman-like key exchange
function key_exchange(sk1, pk2)

    # Generate salt
    keySalt = generate_random_salt(64)  

    # Collect context values with pub_key
    k = collect(codeunits(pk2))
    a_ctx = HMAC_CTX(ctx, k)

    # Perform iterations of CSIDH protocol
    for i in 1:iterations

        # Generate Unit Key from context
        update!(a_ctx, b"$i")
        key = bytes2hex(digest!(a_ctx))
        
        # Generate (rest of) Keys
        sk2 = bytes2hex(hmac_sha3_512(sha3_512(key), bytes2hex(keySalt)))

        pk1 = generate_public_key(sk1)

        # Compute the shared secret
        shared_secret = hash(pk1 .* sk2)

        # Update the secret key
        sk = (shared_secret % p for j in 1:length(sk1))

        # Return the final secret key
        if i == iterations
            return hash(sk)
        end    
    end
end

function validate_chain(chain::Vector{Union{XBlock, CBlock}})
    merkle_tree = build_merkle_tree([block.hash for block in chain])
    
    for i in 2:length(chain)
        current_block = chain[i]
        previous_block = chain[i - 1]
        
        # Check if the previous hash matches
        if current_block.previous_hash != previous_block.hash
            return false
        end
        
        # Verify Merkle tree membership
        if !verify_merkle_membership(merkle_tree, current_block.hash)
            return false
        end
        
        # Validate the individual block
        if !validate_block(current_block, chain[1:i-1])
            return false
        end
    end
    return true
end


function verify_merkle_membership(merkle_tree, transaction_hash)
    current_hash = transaction_hash
    for node in merkle_tree
        current_hash = sha256(current_hash * node.hash)
    end
    return current_hash == merkle_tree[1].hash
end

function start_node_server(node::Node, artificial_chain::Vector{Union{XBlock, CBlock}})
    server = HTTP.Server(socket -> begin
        while true
            req = HTTP.Messages.parse_request(HTTP.Messages.Request(socket))
            res = HTTP.Response("HTTP/1.1", 200, Dict("Content-Type" => "text/plain"), "Received: $(req.body)")
            write(socket, res)
            flush(socket)

            # When a new block is received, validate the entire chain
            if req.body != ""
                new_block = parse_block(req.body)  # Assume you have a function to parse the block
                push!(artificial_chain, new_block)
                
                if validate_chain(artificial_chain)
                    println("Blockchain is valid!")
                else
                    println("Blockchain is invalid!")
                end
            end
        end
    end)
    HTTP.serve(server, node.ip, 8000 + node.id)
end


function validate_block(new_block, artificial_chain)
    if length(artificial_chain) == 0
        # First block in the chain, no need for validation
        return true
    end

    # Validate the new block using Merkle tree and other checks
    merkle_tree = build_merkle_tree([block.hash for block in [new_block]])
    previous_block = artificial_chain[end]
    if new_block.previous_hash == previous_block.hash &&
        verify_merkle_membership(merkle_tree, new_block.hash) &&
        new_block.hash == sha256(new_block)
        return true
    else
        return false
    end
end

# Function to connect a node to the manager
function connect_node(manager::DecentralizedManager, node::Node)
    println("Node $node.id with IP $node.ip connected.")
    # Start the server for node-to-node communication
    Threads.@spawn start_node_server(node)
end

# Extract relevant parts and create a Julia dictionary
function parse_json_to_dict(json_data)
    # Split the JSON data into components
    parts = split(json_data, ',')

    # Extract the relevant information
    hash_part = parts[1]
    dict_part = join(parts[2:end-1], ',')

    # Create a Julia dictionary
    parsed_dict = Dict("hash" => strip(hash_part, "[\"]"), "data" => JSON.parse(dict_part), "number" => parse(Int, parts[end]))
    println(parsed_dict)
    return parsed_dict
end

function parse_request(endpoint, data, client)
    # TODO: PROTOTYPE: Performance Heavy
    # Measure execution time using @time macro
    # @time result = mining_thread()
    # println("Detailed time from @time macro: ", result)

    if occursin("xchange_genesis", endpoint)
        return gensis_Xchange(a, b, p, deg, j_invariant_E1, ctx, data)
    elseif occursin("coin_genesis", endpoint)
        return genesis_mCoin(data)
    elseif occursin("add_mvrx_block", endpoint)
        d = split(data, '"')
        type = d[4] == "mvrx" ? "pos" : "poa" 
        previous_hash = d[8]
        codec = d[12]
        transaction = d[16]
        compute_power = d[20]
        platform_activity = split(split(d[23], ":")[2], "}")[1]
        block = add_mvrx_block(a, b, p, deg, ctx, type, previous_hash, codec, transaction, BigFloat(compute_power), BigFloat(platform_activity), j_invariant_E1)
        return block
    elseif occursin("get_mvrx_block", endpoint)
        return get_mvrx_block(data)
    elseif occursin("burn_mvrx_block", endpoint)
        return burn_mvrx_block(data)
    elseif occursin("mine_mvrc", endpoint)
        return mine_mvrc_block(userid, num_nodes)
    elseif occursin("mine_mvrx", endpoint)
        return mine_mvrx_block(userid, num_nodes)
    elseif occursin("add_mvrx_transaction", endpoint)
        println(data)
        # Extract strings within double quotations using regular expression
        matches = matchall(r"(?<=\")[^\"]+(?=\")", data)
        # Create a Julia dictionary
        parsed_dict = Dict("hash" => matches[1], "data" => matches[2:end-1], "number" => parse(Int, matches[end]))
        return add_mvrx_transaction(data.previous_block, data.transaction, max_size_MB, j_invariant_E1)
    elseif occursin("add_mvrc_transaction", endpoint)
        return add_mvrc_transaction(data.previous_block, data.transaction, reward, max_size_MB)
    elseif occursin("get_mvr_wallet", endpoint)
        response = get_new_wallet(a, b, p, rho)
        println("response: ", response)
        return response
    elseif occursin("get_new_skey", endpoint)
        return get_new_skey(data)
    elseif occursin("coin_purchase", endpoint)
        return get_new_skey(data)
    elseif occursin("coin_sale", endpoint)
        return get_new_skey(data)
    elseif occursin("hash_compute", endpoint)
        return bytes2hex(sha3_512(data))
    else
        return "Invalid endpoint"
    end
end

function gensis_Xchange(i, j, p, deg, j_invariant, ctx, data)
    # Declare statics
    type = "pos"
    codec = "MuvorX_Genesis"

    # Exctract personal message (Nothing here)
    a, b, c, d = split(data, ',', limit=4)

    # Define zeros over field
    z = zero(rand(0, p))

    # Hash zeros for Genesis Hash
    update!(ctx, b"$z")
    genesis_hash = bytes2hex(digest!(ctx))

    # Add Block and return block apex
    apex = add_mvrx_block(i, j, p, deg, ctx, type, genesis_hash, codec, 0, 100, 100, j_invariant)
    return apex
end

function genesis_mCoin()
    # Initialize the blockchain with a genesis block
    genesis_block = CBlock(0, "0", string(now()), ["Property_of_Wakuu_Corporations"], sha3_512("0000000000000000000000000000000000000000000000000000000000000000"), 10000000.0)
    return genesis_block
end

# Function to simulate nodes performing tasks
function simulate_xminer(node::Node)
    while true
        try
            current_nonce = Threads.atomic_add!(node.shared_nonce, 1)  # Atomically increment the shared nonce
            total_nonce, total_challenge, total_response = pow_prover(private_key, current_nonce, ctx)
            schnorr_verifier(public_key, nonce, challenge, response, ctx)
            println("Node $node.id with IP $node.ip mined nonce-$current_nonce")
            return total_nonce, total_challenge, total_response
        catch e
            return "Mining Failed!"
        end
    end
end

# Function to simulate nodes performing tasks
function simulate_cminer(node::Node)
    while true
        try
            current_nonce = Threads.atomic_add!(node.shared_nonce, 1)  # Atomically increment the shared nonce
            difficulty = exponential_difficulty(current_block.nonce, i)
            reward, block = mining_thread(transaction_data, difficulty, i)
            println("Node $node.id with IP $node.ip mined nonce-$current_nonce")
            return reward, block
        catch e
            return "Mining Failed!"
        end
    end
end

function add_user_to_pool(userid, node, manager)
    # Add user-to-node mappings
    add_user_to_node(manager, userid, node)
    # Simulate users starting mining
    node_ip = manager.user_to_node[user_id]
    println("User $user_id is using $node with IP $node_ip to mine.")
end

function xmine(node)
    # Start the decentralized manager task
    total_nonce, total_challenge, total_response = Threads.@spawn simulate_xminer(node)
    # Wait for manager thread to finish
    wait(total_nonce, total_challenge, total_response)
    return total_nonce, total_challenge, total_response
end

function cmine(node)
    # Start the decentralized manager task
    total_nonce, total_challenge, total_response = Threads.@spawn simulate_cminer(node)
    # Wait for manager thread to finish
    wait(total_nonce, total_challenge, total_response)
    return total_nonce, total_challenge, total_response
end

#  intent to mine mvrx
function use_p2p_node(userid, num_nodes)
    # Initialize the decentralized manager with user-to-node mapping
    # Create and connect nodes
    if num_nodes < 59
        i = num_nodes + 1
        manager = DecentralizedManager(Node[], Dict{String, String}())
        manager = create_node(manager, "192.168.0.$i", "ipfs")
        connect_node(manager, node)
        add_user_to_pool(userid, node, manager)
        return manager
    end
end

#  intent to mine mvrx
function mine_mvrx_block(userid, num_nodes)
    # Initialize the decentralized manager with user-to-node mapping
    # Create and connect nodes
    if num_nodes < 59
        i = num_nodes + 1
        manager = DecentralizedManager(Node[], Dict{String, String}())
        manager = create_node(manager, "192.168.0.$i", "mvrx")
        connect_node(manager, node)
        add_user_to_pool(userid, node, manager)
        return manager
    end
end

#  intent to mine mvrc
function mine_mvrc_block(userid, num_nodes)
    # Initialize the decentralized manager with user-to-node mapping
    # Create and connect nodes
    if num_nodes < 59
        i = num_nodes + 1
        manager = DecentralizedManager(Node[], Dict{String, String}())
        new_manager, node = create_node(manager, "192.168.0.$i", "mvrc")
        connect_node(new_manager, node)
        add_user_to_pool(userid, node, new_manager)
        return manager
    end
end

function gcd(a, b)
    while b != 0
        a, b = b, a % b
    end
    return abs(a)
end

# Function to perform an isogeny map
function sidh_isogeny_map(a, b, l, p, x, y, j_invariant)
    # Check that the point is on the curve
    if is_on_curve(a, b, p, x, y)
        # Perform the isogeny map
        x_mapped, y_mapped = morphism(l, p, x, y, j_invariant) 

        # Ensure the result is on the curve
        if is_on_curve(a, b, p, x_mapped, y_mapped)
            return x_mapped, y_mapped
        end
    end
end

# Function to check if a point is on the curve
function is_on_curve(a, b, p, x, y)
    return round((x / a)^2 + (y / b)^2) == 0.0
end

function compute_phi_from_equation(a, b, p, x, y, deg)
    if gcd(x, p) == 1.0 && round(gcd(y, p)) == 0.0
        # Convert degrees to radians
        theta_rad = deg2rad(deg)
        
        # Calculate tan(phi)
        tan_phi = b * sin(theta_rad) / (a * cos(theta_rad))
        
        # Calculate phi using atan
        phi_rad = atan(tan_phi)
        
        # Convert radians to degrees
        phi_deg = rad2deg(phi_rad)
                    
        return phi_deg
    end
end

function morphism(l, p, x, y, j_invariant)  
    # Calculate the morphism
    x_mapped = x + l
    y_mapped = (y - l^2) * j_invariant % p
    
    return x_mapped, y_mapped
end

# Function to calculate the prime l using isogeny mapping
function calculate_prime_l(a, b, p, x, y, phi, j_invariant)
    # Set an upper bound for the prime search
    upper_bound = 1000

    for l in 2:upper_bound
        # Check if l is prime and satisfies the condition
        if isprime(l)
            # Perform isogeny map
            Qx, Qy = sidh_isogeny_map(a, b, l, p, x, y, j_invariant)
            println(round(atan(Qy / Qx), digits=2), "\n\n", -1 * round(j_invariant, digits=2), "\n\n")

            # Check if the mapped point's coordinates match phi
            if round(atan(Qy / Qx)) ≈ -1 * round(j_invariant)
                return l
            end
        end
    end

    error("Prime not found within the specified range")
end

function extgcd(a, b)
    if a == 0
        return (0, 1, b)
    end
    
    x, y, gcd = extgcd(b % a, a)
    return (y - (b ÷ a) * x, x, gcd)
end

function modinv(a, m)
    if a == 0
        error("The modular inverse does not exist.")
    end
    
    x, y, gcd = extgcd(a, m)
    
    if gcd != 1
        error("The modular inverse does not exist.")
    end
    
    return mod(x, m)
end

function double(P::Point, a, b, p)
    # Check if P is the identity element (O)
    if P == O
        return O
    end
    
    # Compute the double of P
    λ = ((3 * P.x^2 + a) * modinv(2 * P.y, p)) % p
    x3 = (λ^2 - 2 * P.x) % p
    y3 = (λ * (P.x - x3) - P.y) % p
    
    return Point(x3, y3)
end

function +(P::Point, Q::Point, a, b, p)
    # Check if P or Q is the identity element (O)
    if P == O
        return Q
    elseif Q == O
        return P
    end
    
    # Check if P and Q are equal
    if P == Q
        return double(P, a, b, p)
    end
    
    # Compute the sum of P and Q
    λ = ((Q.y - P.y) * modinv(Q.x - P.x, p)) % p
    x3 = (λ^2 - P.x - Q.x) % p
    y3 = (λ * (P.x - x3) - P.y) % p
    
    return Point(x3, y3)
end

# Step 2: Find a point (x) of order l
function find_point_of_order_l(E1, l, p, x, y)
    for Z in 5670000000000/5000000  # Iterate over x-coordinates (i.e. number of available $MVRX: 1,134,000)
        try
            # Find new x-coord
            P = y % p  # Construct a field using x
            Q = P % l
            
            # Compute Dot Product ≈ 1 to compare 0
            dot = dot_product([x, Q], [x, y])

            # Return xν, yν
            if iszero(round(Q, sigdigits=0))
                return dot, Q
            end
        catch
            continue
        end
    end
    error("Could not find a point of order $l")
end

# Step 3: Define the isogeny
function radical_isogeny(X, Y, l, deg)
    # Compute the kernel for the isogeny
    kernel = [X * i for i in 1:deg]

    # Define the isogeny map
    return sum([Y + k for k in kernel]), kernel
end

function calculate_proportional_rewards(compute_power, total_compute_power, total_rewards)
    if total_compute_power == 0
        return 0.0
    end
    
    return compute_power / total_compute_power * total_rewards
end

function calculate_rewards(elapsed_time, compute_power, platform_activity)
    # Define weights for each factor
    time_weight = 0.7
    power_weight = 0.2
    activity_weight = 0.1
    
    # Define a decay factor for time-based rewards
    decay_factor = 0.95  # Example: Rewards decrease by 5% for every time unit
    
    # Calculate rewards based on weights and factors
    time_rewards = time_weight * (elapsed_time * (decay_factor ^ elapsed_time))
    power_rewards = power_weight * compute_power
    activity_rewards = activity_weight * platform_activity
    
    total_rewards = time_rewards + power_rewards + activity_rewards
    
    return total_rewards
end

# Function to allocate memory and create a new mvrx block
function allocate_memory_and_create_block(nonce, type, previous_hash, recomputed_commitment, codec, transaction, compute_power, platform_activity, elapsed_time)
    # Allocate memory for the block
    if type == "poa"
        # Aggregrate Block Data
        erp_codec = codec

        # Calculate the hash based on block data (customize as needed)
        block_data = join([type, string(nonce), previous_hash, string(Dates.now(Dates.UTC)), join([transaction], ","), string(reward), string(join(sha3_512(recomputed_commitment), ","))], ",")
        hash = bytes2hex(sha3_512(block_data))

        # Store Timestamp
        timestamp = Dates.now(Dates.UTC)

        # Compute Reward
        reward = calculate_rewards(elapsed_time, compute_power, platform_activity)

        # Populate the fields
        block = XBlock(nonce, previous_hash, erp_codec, timestamp, String[], reward, hash)  
        return block
    elseif type == "pos"
        # Aggregrate Block Data
        erp_codec = codec

        # Store Timestamp
        timestamp = Dates.now(Dates.UTC)
        
        # PROTOTYPE: Create an example KeyReference with key-value pairs
        # block.consensus = Dict(sha3_512(transaction) => [])
        
        # Compute Reward
        reward = calculate_rewards(elapsed_time, compute_power, platform_activity)

        # Calculate the hash based on block data (customize as needed)
        block_data = string(type, string(nonce), previous_hash, string(timestamp), join(transaction, ","), string(reward), string(recomputed_commitment))
        hash = bytes2hex(sha3_512(block_data))
        consensus = String[bytes2hex(sha3_512(string("Muvor ERP Exchange - Genesis Transaction - Tony 'Wakuu' Walker", transaction)))]

        # Populate the fields
        block = XBlock(nonce, erp_codec, previous_hash, timestamp, consensus, hash, reward)
        apex = Dict("nonce" => nonce, "codec" => erp_codec, "previous_hash" => previous_hash, "timestamp" => timestamp, "consensus" => consensus, "hash" => hash, "reward" => reward)
        return block, JSON.json(apex)
    elseif type == "pow"
        # Aggregrate Block Data
        erp_codec = codec
        block_data = join([type, string(nonce), previous_hash, string(Dates.now(Dates.UTC)), join([transaction], ","), string(reward), string(join(sha3_512(recomputed_commitment), ","))], ",")
        hash = bytes2hex(hmac_sha3_512(sha3_512(key), bytes2hex(block_data)))
        reward = calculate_rewards(elapsed_time, compute_power, platform_activity)
        timestamp = Dates.now(Dates.UTC)
        
        # Generate a random nonce
        n = rand(UInt8, nonce)

        # Compute the challenge @ O(n+1)
        update!(ctx, b"$n")
        key = bytes2hex(digest!(ctx))
        block = XBlock(
            # Populate the fields
            nonce,
            previous_hash,
            erp_codec,
            timestamp,

            # Create an example KeyReference with key-value pairs
            [],
            reward,
            
            # Calculate the hash based on block data (customize as needed)
            hash
        )
        return block
    elseif type == "shim"
        # Aggregrate Block Data
        erp_codec = codec
        block_data = join([type, string(nonce), previous_hash, string(Dates.now(Dates.UTC)), join([transaction], ","), string(reward), string(join(sha3_512(recomputed_commitment), ","))], ",")
        hash = hash(block_data)
        reward = calculate_rewards(elapsed_time, compute_power, platform_activity)
        timestamp = Dates.now(Dates.UTC)
        block = XBlock(
            # Populate the fields
            nonce,
            previous_hash,
            erp_codec,
            timestamp,

            # Create an example KeyReference with key-value pairs
            [],
            reward,
            
            # Calculate the hash based on block data (customize as needed)
            hash
        )
        return block
    end
end

# PoS (PoA)
function add_mvrx_block(a, b, p, deg, ctx, type, previous_hash, codec, transaction, compute_power, platform_activity, j_invariant)
    # block incrementation
    x = increment_nonce()
    println(x)
    # Compute Complex Ellipse E1
    y = sqrt((((1 - a * x^5) - (a * x^3) + (a * x)) % p) / (((1 - b * x^5) - (a * x^3) + (b * x)) % p))
    E1 = :(ax^2 + y^2 -> 1 + bx^2y^2)

    # compute phi from field
    phi = compute_phi_from_equation(a, b, p, x, y, deg)
    # E2, phi = compute_radical_isogeny(E1, j_invariant_E1, l)
    
    # find prime l
    l = calculate_prime_l(a, b, p, x, y, phi, j_invariant)
    
    # find tortion P O(l)
    xn, yn = find_point_of_order_l(E1, l, p, x, y)
    
    # Compute the isogeny for this block
    ln, map = radical_isogeny(xn, yn, l, deg)
    
    # FIXME: DP REQUIRES MATCHING MAP LENGTHS This dot product generates a random <vector> 
    # v = dot_product([xn, yn, ln], map)

    # generate key salt from nonce vector over field
    keySalt = generate_random_salt(64)

    # generate keys from E2
    tempsk = generate_secret_key(phi, keySalt, ctx)
    upk = generate_public_key(tempsk)
    usk = key_exchange(tempsk, upk)

    # run zkp alogs
    commitment, challenge, response = schnorr_prover(usk, ctx)
    println(commitment)
    # CREATE & RETURN BLOCK
    # true-false
    if schnorr_verifier(usk, commitment, challenge, response)
        # Measure execution time using @elapsed macro
        recom = usk * response
        elapsed = @elapsed schnorr_prover(usk, ctx)
        time = @elapsed schnorr_verifier(usk, commitment, challenge, response)
        k_t = @elapsed key_exchange(tempsk, upk)
        elapsed_time = elapsed + time + k_t
        block, a = allocate_memory_and_create_block(x, type, previous_hash, recom, codec, transaction, compute_power, platform_activity, elapsed_time)
        # TODO: save block to mongo
        return a
    else return "non-viable block"
    end
end

function get_mvrx_block(data)
    # Implement logic for handling request at endpoint ep_2
    return "Processed ep_2 request with data: $data"
end

function burn_mvrx_block(data)
    # Implement logic for handling request at endpoint ep_2
    return "Processed ep_2 request with data: $data"
end

function add_mvrc_block(data)
    # get prev hash from db
    add_block(prev_hash, data)
end

function add_mvrxc_transaction(previous_block, transaction, reward, max_size_MB)
    add_transaction(previous_block, transaction, reward, max_size_MB)
end

# Function to add a transaction and create a new block if size exceeds 1MB
function add_mvrx_transaction(previous_block, transaction, max_size_MB, j_invariant)
    index = previous_block.index + 1
    previous_hash = previous_block.hash
    new_transaction_data = vcat(previous_block.transaction_data, transaction)

    # Calculate the size of the block
    block_size_MB = sizeof(Base.serialize(new_block)) / (1024^2) # Convert to MB

    if block_size_MB > max_size_MB
        println("Block size exceeds 1MB. Creating a new block...")
        new_block = add_mvrx_block(a, b, p, deg, ctx, type, previous_hash, codec, transaction, compute_power, platform_activity, j_invariant)
        return new_block
    else
        println("Adding transaction to the current block.")
        return new_transaction_data
    end
end

function get_new_wallet(fa, fb, p, field) 
    global global_nonce

    # Increment the nonce by a fraction if it's already been used
    if global_nonce[] % 1 != 0
        global_nonce[] += 0.001  # You can adjust this fraction as needed
    end
    
    # nonce
    x = global_nonce[]

    # some real generator 
    nonce = p % global_nonce[]

    # Generate a random private key as nonced zeros
    private_key = bytes2hex(sha3_512(string(nonce)))

    # Compute y from curve
    numerator = 1 - (fa * x^2)
    denominator = 1 - (fb * x^2)
    
    # println(numerator)
    # Ensure the denominator is non-zero
    if denominator == 0
        error("Denominator is zero, cannot compute square root")
    end
    
    # println(numerator, "\n", denominator)
    # Compute with modulo
    y = sqrt((numerator % p) / (denominator % p))

    # Generate Public key from the field over nonce
    public_key = bytes2hex(sha3_512(string(field) * string((nonce % y) / (y % nonce))))
    # println("pub: ", public_key, "\n", "priv: ", private_key)
    
    # Generate mnemonic from nonce
    mnemonic = sha3_512(string(x*y^nonce%p))

    # Generate address from keys + mnemonic
    dh = key_exchange(private_key, public_key)
    address = bytes2hex(sha3_512(string(hash(dh * mnemonic))))
    
    response = [public_key, private_key, bytes2hex(mnemonic), address]
    # println("res: ", response)

    return response
end

function get_new_skey(data)
    # Implement logic for handling request at endpoint ep_2
    return "Processed ep_2 request with data: $data"
end

# Function to calculate the j-invariant of an elliptic curve
function j_invariant(E::EllipticCurve)
    a = E.a
    b = E.b
    
    numerator = 4 * a^3
    denominator = 4 * a^3 + 27 * b^2
    
    return 1728 * (numerator / denominator)
end

# Function to compute CSIDH radical isogeny [phi]
function CSIDH_compute_radical_isogeny(E1, E2, j_invariant_E1, l, phi, salt, ctx)
    # Perform iterations of the CSIDH-like protocol
    for i in 1:l
        # Generate random secret keys
        sk_Alice = generate_secret_key(phi, salt, ctx)
        sk_Bob = generate_secret_key(phi, salt, ctx)

        # Compute public keys
        pk_Alice = generate_public_key(sk_Alice)
        pk_Bob = generate_public_key(sk_Bob)

        # Compute the isogeny
        phi_Alice = key_exchange(sk_Alice, pk_Bob)
        phi_Bob = key_exchange(sk_Bob, pk_Alice)

        # Apply the isogenies
        E1 = apply_isogeny(E1, phi_Alice)
        E2 = apply_isogeny(E2, phi_Bob)

        # Update the j-invariant for E1
        j_invariant_E1 = j_invariant(E1)
    end

    return j_invariant_E1
end

function handle_client(client::TCPSocket)
    try
        message = readavailable(client)
        if length(message) == 0
            println("no message")
            return
        end

        # println("Received message:\n", message)  # Add this line for debugging

        # println(split(String(message), ";"), "\n\n")

        # Extract ep and message
        endpoint, request_data = split(String(message), ";", limit=2)
        
        # println("Headers:\n", endpoint)

        # Add this to your handle_client function where appropriate
        if startswith(endpoint, "OPTIONS")
            println("keep-alive")
            response_headers = "HTTP/1.1 200 OK\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: PUT, GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n\r\n"
            write(client, response_headers)
            close(client)
            return
        end

        # Extract endpoint from headers or use a different approach based on your request format
        # println("Body:\n", request_data)
        
        # TODO: Prototype socket handler
        # group = ip"228.5.6.7"
        # socket = Sockets.UDPSocket()
        # send(socket, group, 6789, "Hello over IPv4")

        println(endpoint, "\n\n")
        response = parse_request(endpoint, request_data, client)
        # Add these headers to the response
        headers = "Access-Control-Allow-Origin: *\r\nAccess-Control-Request-Method: PUT, GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Methods: PUT, GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 2592000\r\n"

        # Send the headers along with the response
        res = "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json; charset=utf-8\r\n$headers\r\n$response"
        println(res)
        write(client, res)
        close(client)
    catch err
        println("Connection denied from $ip")
        close(client)
        return err
    end
end

function handle_request(stream::TCPSocket)
    request = HTTP.Messages.parse_request(stream)
    
    # Process the request here and generate a response
    
    # Example response: Hello, World!
    response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 13\r\n\r\nHello, World!"
    
    write(stream, response)
    close(stream)
end

# Generate random salt(s)
function generate_random_salt(length)
    return rand(UInt8, length)
end

# Generate a random secret key
function generate_secret_key(phi, keySalt, ctx)
    # Generate Unit Key from context
    update!(ctx, b"$phi")
    key = bytes2hex(digest!(ctx))
    gsk = deriveKeyFromSoftkey(key, keySalt)

    return bytes2hex(gsk)
end

# Generate a public key from a given secret key
function generate_public_key(sk)
    return bytes2hex(sha3_512(sk))
end

# Generate the genesis block
function generate_hash(phi, keySalt, ctx)
    # Create a random secret key
    sk = generate_secret_key(phi, keySalt, ctx)
    
    # Create the corresponding public key
    pk = generate_public_key(sk)
    
    # Create the block with the secret key and public key
    block = Dict("SecretKey" => sk, "PublicKey" => pk)
    
    return block
end

# Perform a key exchange between two blocks
function perform_key_exchange(block1::Dict{String, Any}, block2::Dict{String, Any})
    sk1 = block1["SecretKey"]
    pk2 = block2["PublicKey"]
    final_secret_key = key_exchange(sk1, pk2)
end

# Main function
function communications_hash(phi, prev_block, ctx)    
    # Generate the next block
    virgin_keys = generate_hash(phi, keySalt, ctx)
    println("XBlock: ", virgin_keys)
    
    # Perform a key exchange between the blocks
    final_secret_key = perform_key_exchange(prev_block, virgin_keys)
    println("Final Secret Key: ", final_secret_key)
    return final_secret_key
end

# Main server function
function main(port)
    # Start the server
    server = Sockets.listen(IPv4(0), port)
    println("Server & Firewall listening on port $port...")
    while true
        try
            client = accept(server)
            println("Client connected.")
            ip, port = getpeername(client)
            for addr in ALLOWED_IPS
                if string(ip) == addr
                    println("Connection allowed from $ip")
                    Threads.@spawn handle_client(client)
                else
                    println("Connection denied from $ip")
                    close(client)
                end
            end
        catch err
            close(client)
            return err
        end
    end
end

# Run the main server function
main(port)
