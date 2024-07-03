# AI-Orchestration

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
    >   - Trained and fine-tuned largely on dialogue
    >   - Optimized for multi-turn conversations, but still fine for single-turn responses
    
    >   Other models are Labelled Instruct:
    >   - Trained and fine-tuned on instruction response pairs
    >   - Not optimized for multi-turn conversations, but may still be capable.

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

3. LangChain

> LangChain Developers encourage using it as a wrapper around an LLM.
- Install langchain-openai library

        $ pip install langchain-openai

-  Check the documentation for LangChain to see the list of supported LLMs
- Prompt Templates: Fills in parts of the promptfor the LLM
- Construct a chain using LangChain Expression Language
    > 
    >       $ cat readme.txt | wc
    > 
    > Similar to this command

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"user_input": user_input})

> **Inspecting the Inputs and Outputs**
>   
>       print(prompt.output_shema)
>       print(llm.input_schema)
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

- **Beware the \*\*kwargs**

    Both langchain llamaindex make use of \*\*kwargs arguments, where a Python dictionary is used instead of strongly typed arguments

## Combine LLMs and Indexes to Query Local Documents

LLMs are trained on billions of documents, but zero of them are your local private files.

The previous chapter mentioned RAG, or retrieval-augmented generation, which is a core technique in expanding the external reach of LLMs. But first, let's look at two alternate approaches and why they have massive downsides. 


- One approach is simply using longer prompts. 

   If you can include the full text of your documents in the prompt, the AI can access all that information to formulate a response. 
   Well, how much text can a prompt hold, anyway? 
   That boils down to something called a context window with a length measured in tokens. A token is a word or a part of a word, and it's the fundamental unit of language that LLMs process.
   The key limitation with this approach, called document stuffing, is that the context window has a limited size. Different language models have different context window sizes.


- The second option of fine-tuning a model. 

   This is a process that adjusts the weights of a language model, or in some cases, it adds entirely new layers of weights based on additional training with additional documents. Typically, this requires thousands of sample documents and lots of GPU muscle to accomplish the retraining. And even with this effort, fine-tuning is better at reaffirming what the model already "knows" as opposed to introducing genuinely new knowledge.

Hence we use RAG 

1. Choosing an embedding:

   Some embeddings can also handle images, which means that it's possible to quickly match up text with similar images.
   The shape of your data will determine what kinds of embeddings will work best for you. For instance, if you need to compare words and images, you should choose an embedding that encodes both words and images. Other embeddings include geographic information in the vector so that physical proximity factors into the similarity measure. And for ordinary text similarity use cases, there are a number of embedding models that will perform well.
   After accounting for the shape of your data, possibly the biggest decision point is whether to produce your embeddings with a cloud model or something run locally. 

   **Local Embedding**:
   - Better for privacy
   - You are responsible for operations
   - Ability to run without internet access

   **Cloud Embedding**:
   - Potential privacy leak
   - Outsourced operations
   - If service is down, so is your app

    Some Embeddings:

    1. OpenAI Embedding: ada v2
       Name: text-embedding-ada-002
       Dimensions: 1536
       Recommended chunk size: 256 or 512 (max 8191)
    
    2. LangChain and LLamaIndex Embeddings
       As of this recording, LangChain defaults to one from the sentence transformers project. But for anything beyond quick and dirty experiments, you shouldn't rely on defaults. Defaults can change out from under you at any time. LangChain offers support for some 40 different text embedding providers, so there's plenty of room to experiment.

    3. BAAI Embedding: bge-small-en
       Name: BAAI/bge-small-en
       Dimensions: 384
       Recommended chunk size: 512 tokens
       _Available on huggingface_

    **Embedding Tips**:
    
    - Break documents into small chunks, often 128,256 or 512 tokens

      Different embedding models will have a different preferred chunk size.  Really large chunks of text aren't good because they can get computationally expensive, and returning huge chunks will quickly fill up if not overflow your LLMs context window. 
      But really small chunks aren't great either. You can imagine that in chopping up, say, a sentence into small pieces, no single piece conveys much meaning, so you lose specificity.
      In general, it's best to use the smallest chunk size that doesn't lose context.
    
    - Split on natural boundaries, like headings, if possible

    - Otherwise, include some overlap between chunks

    - Keep in mind the context window size of your LLM; a RAG app might include 1-10 chunks in the prompt.

    ***LlamaIndex was practically built around the use case of RAG.***

2. RAG with LLamaIndex

    LlamaIndex is really in its happy place when we're indexing documents.

        index = VectorStoreIndex(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query("Who was Shakespeare's wife?")
    
    Instead of calling query_engine.query, we could have also called as chat engine. 
    You want to be able to choose your own embedding and LLM. So we're going to use a longer form here, one that lets us plug in different pieces in a service context.


    1. Set up <mark>ServiceContext</mark> and embed model

       A **service context** is like a switchboard operator for AI telling our app what to use for different components, including embedding, embedding store, or even the LLM itself. We'll use this feature to plug in a local embedding chosen from Hugging Face, as well as continue to use the locally served model-compatible with the OpenAI API. 
        ```
        from llama_index.core import (load_index_from_storage, 
                                        set_global_service_context)
        from llama_index import ServiceContext
        service_context = ServiceContext.from_defaults(
            llm=llm,
            chunk_size=512,
            chunk_overlap=64,
            embed_model=embed_model
        )
        set_global_service_context(service_context)
        ```
       But latest code is to do it via **Settings**
        ```
        from llama_index.core import ( Settings,
                                        load_index_from_storage)
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        Settings.embed_model = embed_model
        ```
    2. Use <mark>SimpleDirectoryReader</mark> or another reader to obtain documents.

       When it comes to reading the documents, we use **SimpleDirectoryReader**. It follows the chunk size and chunk overlap settings, and it supports a huge range of data formats, including _CSV, Microsoft Word, Jupyter Notebooks, PDF, PowerPoint, Markdown, and others_.
        ```
        documents = SimpleDirectoryReader(args.docs_dir).load_data()
        ```
    3. Populate <mark>VectorStoreIndex</mark>

       The default **vector store** will hold all the resulting document chunks.
        ```
        vector_store = VectorStoreIndex.from_documents(documents)
        ```
    4. Create <mark>retriever</mark>

       Once we have a vector store set up, we can create a **retriever** 
        ```
        retriever = VectorIndexRetriever(vector_store)
        query_engine = RetrieverQueryEngine.from_args(
                            retriever=retriever
                        )
        ```
       and from that, create a context chat engine. 
        ```

    5. Create <mark>ContextChatEngine</mark>

        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            query_engine=query_engine,
            verbose=True
        )
        ```
    With one line of code, we can call chat_repl, that's read eval print loop, 
    ```
    chat_engine.chat_repl()
    ```
    and that gives us a bare-bones chat interface. In other words, now we're into interactive conversation mode.

    >   In order to work with the Hugging Face embeddings and to read Microsoft Word files, we're going to need a few more installed modules. These modules don't appear in our code, but are imported from within LlamaIndex code.
    >```
    >pip install docx2txt
    >pip install transformers
    >pip install torch
    >```

    ***Go through LLamaIndex Documentation or LLama Hub***
    
    We're using simple directory reader to read local files, but beyond just documents, you can also talk to GitHub or Wikipedia, Jira databases, and lots more. 
    
    > One quick warning for this code. What you see here does not have any change detection in it. So once the documents are indexed, that's what's in the index. And even if the documents change, it's not going to go back and reindex them.