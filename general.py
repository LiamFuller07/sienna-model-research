import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the conversations dataset
df = pd.read_csv('C:/Users/busin/Downloads/updated_conversations.csv')

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode all conversations and save embeddings
conversation_embeddings = model.encode(df['Conversation'].tolist(), show_progress_bar=True)
pd.DataFrame(conversation_embeddings).to_csv('C:/Users/busin\Downloads/conversation_embeddings_psa.csv', index=False)

# Example queries
queries = [ 
    "Hey I have some exams coming up can you give me some study tips for effective time management during exams?",
    "I've been feeling really sick lately can you recommend me some options for a balanced diet?",
    "I'm having an issue with my phone not taking any battery charge do you know how I can fix it?",
    "Can you suggest some interesting books to read this weekend?",
    "I want to go on a vacation in a few weeks, can you recommend me some memorable places to stay?",
    "Can you recommend me a workout routine, I've been feeling very lazy today",
    "What is Elon Musk doing with the new Starship to Mars?",
    "Can you suggest ways I can manage my stress? I've been having really bad sleeps lately",
    "What Christmas presents should I buy for my friend?",
    "I need to put together some New Year's resolutions, can you help me?"
]

# Encode the queries and save embeddings
query_embeddings = model.encode(queries, show_progress_bar=True)
pd.DataFrame(query_embeddings).to_csv('C:/Users/busin/Downloads/query_embeddings_psa.csv', index=False)

# Set the number of top results to retrieve
top_k = 3

# Data structure to store results
results = []

# Perform similarity search for each query
for i, query_embedding in enumerate(query_embeddings):
    query = queries[i]
    similarities = cosine_similarity([query_embedding], conversation_embeddings)
    most_similar_indexes = similarities.argsort()[0][::-1][:top_k]

    # Store the top k most similar conversations for this query
    for rank, index in enumerate(most_similar_indexes):
        result = {
            'Query': query,
            'Rank': rank+1,
            'Conversation': df['Conversation'].iloc[index],
            'Similarity Score': similarities[0][index]
        }
        results.append(result)

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('C:/Users/busin/Downloads/similaritysearch_psa.csv', index=False)
