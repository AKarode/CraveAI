import openai

class GPTIntegration:
    def __init__(self, api_key):
        """
        Initializes the GPT-4 integration class with the provided API key.
        
        Args:
            api_key (str): The OpenAI API key.
        """
        openai.api_key = api_key

    def generate_response(self, prompt, model="gpt-4", max_tokens=150):
        """
        Generates a response from the GPT-4 model based on the provided prompt.
        
        Args:
            prompt (str): The input prompt for GPT-4.
            model (str): The GPT model to use (e.g., "gpt-4").
            max_tokens (int): Maximum number of tokens in the response.
            
        Returns:
            str: The generated response from GPT-4.
        """
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return response.choices[0].text.strip()

    def refine_query(self, user_query, similar_items):
        """
        Refines a user's query by asking GPT-4 to make a recommendation based on similar items.
        
        Args:
            user_query (str): The user's original query.
            similar_items (list): A list of similar items to consider.
            
        Returns:
            str: The refined query or recommendation.
        """
        prompt = (
            f"User query: {user_query}\n"
            f"Here are some similar items: {', '.join(similar_items)}.\n"
            "Based on this, what would you recommend?"
        )
        return self.generate_response(prompt)
