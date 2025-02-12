from openai import OpenAI

client = OpenAI(api_key='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')

def test_openai_api_key():
    # Set your OpenAI API key here

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",  # Using a chat model
        messages=[
            {"role": "user", "content": "The quick brown fox jumps over the lazy"}
        ])
        print("API Key is working. Response from OpenAI:")
        print(response.choices[0].message.content)  # Print the text completion result
    except Exception as e:
        print("Failed to use OpenAI API. Error:")
        print(str(e))

if __name__ == "__main__":
    test_openai_api_key()
