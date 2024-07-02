# AI-Orchestration
***
My exploration of LLMs with Lang Chain and LLaMa-Index

### Basic Installation Setup

The following commands set up the virtual environment

    $ python3.11 -m venv .venv
    $ source .venv/bin/activate

Installing the necessary libraries

    (.venv) $ pip3 install openai
    (.venv) $ pip3 install langchain
    (.venv) $ pip3 install llama-index

To update the outdated packages

    (.venv) $ pip3 install --upgrade (name of the package)

Get the OpenAI API key from: https://platform.openai.com/api-keys

## Building the First App
---
A software tool to:
- Help coordinate the overall operation of AI applications
- Hide the vendor-specifi details and boilerplate code needed to talk to the APIs

Two most popular frameworks to do this are:
- LangChain
- LLamaIndex
With orchestration frameworks, it will be easy to try out newer models and to make changes in the future

The first App:
    The prompt will consist of two parts:
    - pre-written section: Part of the app
    - user-supplied section: Reflects user input
    Combine the two, send prompt to LLM, get response

    1. OpenAI
        1. Get the OpenAI API key
        >   ChatGPT is a Chat model: 
        >       - Trained and fine-tuned largely on dialogue
        >       - Optimized for multi-turn conversations, but still fine for single-turn responses
        
        >   Other models are Labelled Instruct:
        >       - Trained and fine-tuned on instruction response pairs
        >       - Not optimized for multi-turn conversations, but may still be capable.

    2. Running Local LLMs
        1. Install the LM Studio: Desktop app 
        2. Search and Download "una cybertron" model at Q6_K quantization
        3. Locate the model card for this model; find the Prompt template setting
        4. Load the model into memory and examine LM Studio settings.
        5. Chat with the model to get a feel for it
        6. Download a second model with different quantization level and compare
        We have to give the base URL in the file while initializing our openai object and also make sure that we started the server
        > **Not for Production Use!**
        > - LM Studio is great for developement and experimentation, but it's not bulletproof
        > - To serve a local model in production, you'll need a different inference server

    3.LangChain
    > LangChain Developers encourage using it as a wrapper around an LLM.
        - Install langchain-openai library

            pip install langchain-openai
        
        -  Check the documentation for LangChain to see the list of supported LLMs
        - Prompt Templates: Fills in parts of the promptfor the LLM
        - Construct a chain using LangChain Expression Language
            > 
            >   cat readme.txt | wc
            > 
            > Similar to this command

            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"user_input": user_input})
        
        > **Inspecting the Inputs and Outputs**
        >   
        >   print(prompt.output_shema)
        >   print(llm.input_schema)
        >
    
        - for streaming

            result = chain.stream({"user_input": user_input})
            for chunk in result:
                print(chunk, end=' ')
        
    

