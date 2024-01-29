using Random
using SHA
using Base64

# Generate a secure random number
function generateRandomNumber()
    rng = RandomDevice()
    rand(rng, UInt32)
end

# PBKDF2-HMAC-SHA256 key derivation
function pbkdf2_hmac_sha256(softkey, salt)
    key = hmac_sha3_512(sha3_512(softkey), bytes2hex(salt))
    return key
end

# Key derivation function 
function deriveKeyFromSoftkey(softkey, salt)
    key = pbkdf2_hmac_sha256(softkey, salt)
    return key
end

# Example usage:
# randomNum = generateRandomNumber()
# println("Secure random number:", randomNum)

# password = "mySecretPassword"
# salt = [0x01, 0x02, 0x03, 0x04]  # Replace with a random salt
# iterations = 10000  # Adjust as needed
# keyLength = 64  # Key length in bytes

# derivedKey = deriveKeyFromPassword(password, salt, iterations, keyLength)
# println("Derived key:", Base64.encode(derivedKey))
