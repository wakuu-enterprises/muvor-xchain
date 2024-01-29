import Pkg 

# A library for algebraic number theory
using Nemo

struct EllipticCurve
    a::fpFieldElem
    b::fpFieldElem 
    field::Nemo.fpField
end

function EllipticCurve(a, b, field)
    # Check if the discriminant is non-zero
    @assert 4a^3 + 27b^2 ≠ 0
    
    # Check that the field type matches the type of a and b
    @assert typeof(a) == typeof(b) == field
    
    return EllipticCurve(a, b, field)
end

struct Isogeny
    phi
end

function Isogeny(phi)
    return Isogeny(phi)
end

function PR(F, var::Union{Char, AbstractString})
    R, x = PolynomialRing(FiniteField(p), var)
    return R, x
end

function QuadraticExtension(p, q)
    # Define the polynomial ring R = Fp[x]
    R, x = PR(q, 'x')
    x0 = 
    u = im
    # Define the quotient ring Q = R/(x^2 - poly)
    Q = ResidueRing(R, x0 * x0 - u * u + 1)
    
    return Q, x, Fp
end

# Define the field extension for the quadratic extension
# u = 2
# Q, x, Fp = QuadraticExtension(p, u * u + 1)

function generate_weierstrass_model(p, A, B)
    F = FiniteField(p)
    R, x = PolynomialRing(F, "x")
    A = coerce(F, A)
    B = coerce(F, B)
    curve_eq = x^3 + A*x + B
    return curve_eq
end


# Function to compute radical isogeny
function compute_radical_isogeny(E1, j_invariant_E1, l)
    # Define the quadratic twist parameters
    D = a % p
    D2 = D * EF(p)
    A2 = D2
    B2 = one(p)
    F = EF(p)
    
    # Define the supersingular elliptic curve E2
    E2 = EllipticCurve([FiniteField(A2), FiniteField(B2)])
    
    return E2
end

# Function to apply an isogeny to an elliptic curve
function apply_isogeny(E::EllipticCurve, phi::Isogeny)
    # In this simplified example, we'll multiply the x-coordinate of points by n
    E_transformed = deepcopy(E)  # Create a copy of the curve
    
    # Apply the isogeny by multiplying x coordinates by n
    for point in E_transformed
        point[1] *= phi.n  # Assuming point is represented as [x, y]
    end
    
    return E_transformed
end

# # Sample usage
# E2, phi = compute_radical_isogeny(a, b, j_invariant_E1, l)

# # Output the coefficients of the new curve E2: y^2 = x^3 + A * x + B
# A, B = coefficients(E2)
# println("Coefficients of E2: A = $A, B = $B")