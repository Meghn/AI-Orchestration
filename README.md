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

    4. LLama-Index
        LLama-Index has a different philosophy than LangChain about how to structure AI apps
        
        _Basic AI Querying_
        
        **Embedding** : A way to represent text that's more convenient for computers
        LM Studio doesn't serve the embedding API, so this one will be dependent on talking to OpenAI and won't run 100 percent locally.

        We use a LLama-Index construct called a vector store index
        
        **Vector Store** : A vector store is a database that stores documents and their embeddings, specially made for retrieval, based on semantic similarity.
        
        Loading the documents into the index requires embedding.  And by default, it's using the OpenAI API key environment variable to call out to the OpenAI embeddings API and then storing the results.

        In helloqa_llamaindex.py, the query also will get sent off to the embedding server and turned into an embedding. And then the vector store will compare these embeddings.

        >  So this is the effective prompt that the LLM sees based on the operations of LlamaIndex and the vector store. So it starts off with a system prompt. Says, "You are an expert QA system" and has some rules. And then it says, "Context information is below." So below this dotted line, this gets filled in by what gets returned from the vector store. This is based on semantic similarity. So we can see of the three facts, it decided that only two of them were potentially relevant. It included these two sentences in the context information that was part of the prompt. And then it said, "Given the context information and not prior knowledge, answer the query: Who was Shakespeare's wife?" And the answer is what we will get when we run the app. 

        This is called the ***Retrieval-Augmented Generation (RAG)***: Technique for combining data retrieval with an LLM processing to improve quality, accuracy, and verifiability of responses from an AI application

    - **Debugging LangChain**

        from langchain.globals import set_debug, set_verbose
        set_debug(True)
        set_verbose(True)

    - **Debugging LLama-Index**
        - Works well with standard Python logging
        - Various functions and constructors accept a verbose=True parameter

    - **Beware the \*\*kwargs **
        Both langchain llamaindex make use of \*\*kwargs arguments, where a Python dictionary is used instead of strongly typed arguments

## Combine LLMs and Indexes to Query Local Documents
___

