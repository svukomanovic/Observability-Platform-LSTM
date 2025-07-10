import numpy as np

def get_vector_for_log(log_tokens, model, vector_size=100):
    """
    Creates a single feature vector for a log message by averaging its word vectors.
    """
    # Create a list of vectors for the words in the log that are in the model's vocabulary
    vectors = [model.wv[word] for word in log_tokens if word in model.wv]
    
    if len(vectors) == 0:
        # If no words are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    
    # Average the vectors to get a single vector for the entire log message
    return np.mean(vectors, axis=0)

# --- Example of creating a feature vector for a new log message ---

new_log = "Error: Back-off pulling image for pod my-app-xyz"
print(f"\n--- Creating Feature Vector for a New Log ---")
print(f"Original Log: '{new_log}'")

# 1. Pre-process the new log
new_log_tokens = preprocess_log_message(new_log)
print(f"Processed Tokens: {new_log_tokens}")

# 2. Convert to a single vector
log_feature_vector = get_vector_for_log(new_log_tokens, model)
print(f"Resulting Feature Vector (first 10 dims):\n{log_feature_vector[:10]}")