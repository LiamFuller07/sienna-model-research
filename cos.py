import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the conversations dataset
df = pd.read_csv('C:/Users/busin/Downloads/updated_conversations.csv')

# Encode all conversations
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
conversation_embeddings = model.encode(df['Conversation'].tolist(), show_progress_bar=True)

# Example queries for testing
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

# Encode the queries
query_embeddings = model.encode(queries, show_progress_bar=True)

# Set the number of top results to retrieve
top_k = 5

# Perform similarity search for each query
for i, query_embedding in enumerate(query_embeddings):
    query = queries[i]
    similarities = cosine_similarity([query_embedding], conversation_embeddings)
    most_similar_indexes = similarities.argsort()[0][::-1][:top_k]  # Get the indexes of the top k most similar conversations

    # Display the top k most similar conversations for this query
    print(f"Query: {query}")
    for rank, index in enumerate(most_similar_indexes):
        print(f"Rank {rank+1}:")
        print(f"Conversation: {df['Conversation'].iloc[index]}")
        print(f"Similarity Score: {similarities[0][index]}\n")
