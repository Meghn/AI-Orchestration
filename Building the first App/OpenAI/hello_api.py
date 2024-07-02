import os
from openai import OpenAI # version 1.0+
# pip install --upgrade openai

llm = OpenAI(
    # place your OpenAI key in an environment variable
    api_key=os.environ['OPENAI_API_KEY'], # this is the default
    #base_url="http://localhost:1234/v1"  # see chapter 1 video 3
)

system_prompt = """Given the following short description
    of a particular topic, write 3 attention-grabbing headlines 
    for a blog post. Reply with only the titles, one on each line,
    with no additional text.
    DESCRIPTION:
"""
user_input = """AI Orchestration with LangChain and LlamaIndex
    keywords: Generative AI, applications, LLM, chatbot"""

# Call the API and get the response
response = llm.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens=500, # maximum number of tokens we want returned
    temperature=0.7, # 0 gives same response everytime, closer to 1 more random
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
)

print(response.choices[0].message.content)