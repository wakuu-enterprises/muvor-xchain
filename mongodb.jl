using Pkg
using Mongoc
# Connect to MongoDB
# client = connect_to_mongodb()

# Access a specific database
# db = Mongoc.Database(client, "test")

# Access a specific collection within the database
# collection = Mongoc.Collection(db, "your_collection_name")

# Connect to the MongoDB server
function connect_to_mongodb()
    client = Mongoc.Client()
    return client
end

# Disconnect from the MongoDB server
function disconnect_from_mongodb(client::Mongoc.Client)
    Mongoc.close(client)
end

# Get a specific database
function get_database(client::Mongoc.Client, database_name::AbstractString)
    return Mongoc.Database(client, database_name)
end

# Get a specific collection within a database
function get_collection(database::Mongoc.Database, collection_name::AbstractString)
    return Mongoc.Collection(database, collection_name)
end

# Insert a document into a collection
function insert_document(collection::Mongoc.Collection, document::Dict)
    Mongoc.insert!(collection, document)
end

# Find documents in a collection based on a query
function find_documents(collection::Mongoc.Collection, query::Dict)
    return Mongoc.query(collection, query)
end

# Update documents in a collection based on a query and update specification
function update_documents(collection::Mongoc.Collection, query::Dict, update::Dict)
    Mongoc.update!(collection, query, update)
end

# # Perform CRUD operations
# # For example, to insert a document
# document = Dict("name" => "John", "age" => 30)
# Mongoc.insert!(collection, document)

# # Insert a document
# document = Dict("name" => "John", "age" => 30)
# insert_document(collection, document)

# # Query the collection
# query = Dict("name" => "John")
# results = Mongoc.query(collection, query)

# # Find documents matching a query
# query = Dict("name" => "John")
# results = find_documents(collection, query)

# # Iterate over the results
# for result in results
#     println(result)
# end

# # Iterate over the results
# for result in results
#     println(result)
# end

# # Update a document
# update = Dict("\$set" => Dict("age" => 31))
# Mongoc.update!(collection, query, update)

# # Update documents matching a query
# update = Dict("\$set" => Dict("age" => 31))
# update_documents(collection, query, update)

# Delete documents in a collection based on a query
function delete_documents(collection::Mongoc.Collection, query::Dict)
    Mongoc.delete!(collection, query)
end

# # Delete documents matching a query
# delete_documents(collection, query)

# # Delete a document
# Mongoc.delete!(collection, query)

# # Disconnect from MongoDB
# disconnect_from_mongodb(client)