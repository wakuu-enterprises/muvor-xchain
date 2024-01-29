using SHA
using Dates
using Base.Threads

# Define a custom type for signatures
# struct Signature
#     data::Vector{String}
# end

# Define a block structure
struct CBlock
    nonce::Int
    previous_hash::String
    timestamp::String
    transaction_data::Vector{String}  # Change data to hold an array of transactions
    hash::String
    reward::Float64
end

# Function to calculate the reward for a block
function calculate_reward(index)
    50 / 2^index  # Halving every 2 blocks
end

# Function to calculate the hash of a block
function calculate_hash(index, previous_hash, timestamp, data)
    sha256("$index$previous_hash$timestamp$data")
end

# Function to create a new block
function create_block(index, previous_hash, data, reward)
    timestamp = string(now())
    hash = calculate_hash(index, previous_hash, timestamp, data)
    CBlock(index, previous_hash, timestamp, data, hash, reward)
end

# Function to add a transaction and create a new block if size exceeds 1MB
function add_transaction(previous_block, transaction, reward, max_size_MB)
    index = previous_block.index + 1
    previous_hash = previous_block.hash
    new_transaction_data = vcat(previous_block.transaction_data, transaction)
    new_block = create_block(index, previous_hash, new_transaction_data, reward)

    # Calculate the size of the block
    block_size_MB = sizeof(Base.serialize(new_block)) / (1024^2) # Convert to MB

    if block_size_MB > max_size_MB
        println("Block size exceeds 1MB. Creating a new block...")
        return new_block
    else
        println("Adding transaction to the current block.")
        return new_block
    end
end

# Function to mine a new block
function mine_block(previous_block, data, reward, difficulty)
    index = previous_block.index + 1
    previous_hash = previous_block.hash
    new_block = create_block(index, previous_hash, data, reward)

    # Mining loop with exponential difficulty
    while true
        new_block.hash = calculate_hash(new_block.index, new_block.previous_hash, new_block.timestamp, new_block.data)
        if check_difficulty(new_block.hash, difficulty)
            println("CBlock mined! Hash: ", new_block.hash)
            return new_block
        end
    end
end

# nonce::Int
# previous_hash::String
# timestamp::String
# transaction_data::Vector{String}  # Change data to hold an array of transactions
# hash::String
# reward::Float64

# Define mining difficulty parameters
# difficulty_factor = 2.0
# difficulty = 1

# Function to check if a block's hash meets the difficulty criteria
function check_difficulty(hash, difficulty)
    count_zeros = 0
    for char in hash
        if char == '0'
            count_zeros += 1
        else
            break
        end
    end
    count_zeros >= difficulty
end

# Mining loop for a thread
function mining_thread(data, difficulty, thread_id)
    println("Thread $thread_id started mining...")
    previous_block = last(blockchain)
    reward = calculate_reward(previous_block.index)
    new_block = mine_block(previous_block, data, reward, difficulty)
    push!(blockchain, new_block)
    println("Thread $thread_id finished mining.")
    return reward, block
end

# Define an exponential difficulty function
function exponential_difficulty(difficulty_factor, current_block_index)
    difficulty = Int(difficulty_factor^current_block_index)
    return difficulty
end


# # Define the number of blocks to mine and the data for transactions
# num_blocks = 5
# transaction_data = "Transaction Data"

# # Start mining threads
# threads = []
# for i in 1:num_blocks
#     push!(threads, Threads.@spawn mining_thread(transaction_data, difficulty, i))
#     difficulty = exponential_difficulty(current_block.nonce, i)
# end

# # Wait for all threads to finish
# for thread in threads
#     wait(thread)
# end

# # Print the blockchain
# for block in blockchain
#     println("CBlock Index: ", block.index)
#     println("Previous Hash: ", block.previous_hash)
#     println("Timestamp: ", block.timestamp)
#     println("Data: ", block.data)
#     println("Hash: ", block.hash)
#     println("Reward: ", block.reward)
#     println()
# end
