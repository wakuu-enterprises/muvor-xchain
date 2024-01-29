using Pkg
using SHA
using Random
using OpenSSL
using Elliptic

function createWallet(public_key)
    # Apply SHA-256 hash to the public key
    hashed_public_key = sha256(string(public_key))

    # Add a version byte (optional)
    if mainnet
        return version_byte = 0x42
    elseif testnet
        return version_byte = 102
    end
    hashed_public_key = version_byte * hashed_public_key

    # Add a checksum (optional)
    checksum = sha256(sha256(hashed_public_key))[1:4]
    hashed_public_key = string(hashed_public_key, checksum)

    # Base58 encode
    return wallet_address = base58encode(BigInt(hashed_public_key))
end

function dot_product(vector1, vector2)
    if length(vector1) != length(vector2)
        throw(ArgumentError("Vectors must have the same length"))
    end

    result = 0
    for i in 1:length(vector1)
        result += vector1[i] * vector2[i]
    end

    return result
end

# PoS Prover's side
function schnorr_prover(private_key, ctx)
    # Generate a random nonce
    n = rand(UInt8, 1)

    # Compute the challenge @ O(n+1)
    update!(ctx, b"$private_key")
    key = bytes2hex(digest!(ctx))
    challenge = hmac_sha3_512(sha3_512(key), bytes2hex(n))

    # Compute the response
    response = (dot_product(sha3_512(key), sha3_512(challenge)) + private_key) % private_key
    return key, challenge, response
end

# PoW Prover's side with a pool of users
function pow_prover(pool_private_keys, nonce, ctx)
    num_users = length(pool_private_keys)

    # Initialize accumulators for nonce, challenge, and response
    total_nonce = UInt256(0)
    total_challenge = UInt256(0)
    total_response = UInt256(0)
    # Generate nonces, compute challenges, and responses for each user
    for private_key in pool_private_keys        
        # from potential block
        challenge = generate_challenge(nonce)
        response = (nonce + private_key * challenge) % UInt256(SHA256(challenge))

        # Accumulate values
        total_nonce += nonce
        total_challenge += challenge
        total_response += response
    end

    return total_nonce, total_challenge, total_response
end

# Function to generate a challenge with increasing computational complexity
function generate_challenge(n)
    salt = rand(UInt8, 16)
    
    t_cost = 2^n  # Increase the number of iterations exponentially
    m_cost = 2^(16+n)  # Increase the memory cost exponentially
    parallelism = 1  # Keep parallelism at 1
    
    argon = argon2_hash("Challenge Data", salt, t_cost, m_cost, parallelism)
    hash = scrypt(ScryptParameters(t_cost, m_cost, parallelism), argon, salt, 64) # oops...
    return hash
end

# Verifier's side
function schnorr_verifier(schnorr_key, commitment, challenge, response)
    # Recompute the commitment point
    recomputed_commitment = schnorr_key * response

    # Verify that the commitment point matches the received commitment
    if recomputed_commitment == ((dot_product(sha3_512(commitment), sha3_512(challenge)) + schnorr_key) % schnorr_key) * schnorr_key
        return true
    else
        return false
    end
end

# # Prover generates proof
# nonce, challenge, response = schnorr_prover(private_key, ctx)

# # Verifier verifies the proof
# result = schnorr_verifier(public_key, nonce, challenge, response, ctx)

# if result
#     println("Proof verified successfully.")
# else
#     println("Proof verification failed.")
# end