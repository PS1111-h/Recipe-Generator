# Install the OpenAI SDK first: `pip3 install openai`

import openai

# Set up the OpenAI client with the DeepSeek API endpoint and your API key
openai.api_key = "sk-809cacc753674c8e912dcac8c1f784a1"  # Replace with your actual API key
openai.api_base = "https://api.deepseek.com"  # DeepSeek API endpoint

# Make the API request
response = openai.ChatCompletion.create(
    model="deepseek-chat",  # Specify the model
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False  # Set to True if you want to stream the response
)

# Print the assistant's response
print(response['choices'][0]['message']['content'])