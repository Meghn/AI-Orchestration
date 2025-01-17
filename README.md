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
    ```python
        index = VectorStoreIndex(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query("Who was Shakespeare's wife?")
    ```

    Instead of calling query_engine.query, we could have also called as chat engine. 
    You want to be able to choose your own embedding and LLM. So we're going to use a longer form here, one that lets us plug in different pieces in a service context.


    1. Set up <mark>ServiceContext</mark> and embed model

       A **service context** is like a switchboard operator for AI telling our app what to use for different components, including embedding, embedding store, or even the LLM itself. We'll use this feature to plug in a local embedding chosen from Hugging Face, as well as continue to use the locally served model-compatible with the OpenAI API. 
        ```python
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
        ```python
        from llama_index.core import ( Settings,
                                        load_index_from_storage)
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        Settings.embed_model = embed_model
        ```
    2. Use <mark>SimpleDirectoryReader</mark> or another reader to obtain documents.

       When it comes to reading the documents, we use **SimpleDirectoryReader**. It follows the chunk size and chunk overlap settings, and it supports a huge range of data formats, including _CSV, Microsoft Word, Jupyter Notebooks, PDF, PowerPoint, Markdown, and others_.
        ```python
        documents = SimpleDirectoryReader(args.docs_dir).load_data()
        ```
    3. Populate <mark>VectorStoreIndex</mark>

       The default **vector store** will hold all the resulting document chunks.
        ```python
        vector_store = VectorStoreIndex.from_documents(documents)
        ```
    4. Create <mark>retriever</mark>

       Once we have a vector store set up, we can create a **retriever** 
        ```python
        retriever = VectorIndexRetriever(vector_store)
        query_engine = RetrieverQueryEngine.from_args(
                            retriever=retriever
                        )
        ```
       and from that, create a context chat engine. 

    5. Create <mark>ContextChatEngine</mark>
       ```python
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            query_engine=query_engine,
            verbose=True
        )
       ```
    
    With one line of code, we can call chat_repl, that's read eval print loop, 
    ```python
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

3. RAG with LangChain
   
   There are a lot of ways to accomplish the use case of RAG with LangChain, But the Zen of Python says there should be one and preferably only one obvious way to do it. But LangChain gives us options.

   Naturally we'll use a chain, pre-built one instead of constructing it from LCEL. 

   The structure of our LAngChain app looks a lot like the LLamaIndex one we previously built. One difference is that we're using **sentence transformer embeddings**. 
   ```python
    pip install sentence_transformers
    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
   ```
   And we're specifying an explicit **vector store** using **FAISS** for meta.
   ```python
    vectorstore = FAISS.from_documents(frags, embedding)
   ```
   A class called **Recursive Character Text Splitter** chops up our documents. Play with the chunk size and chunk overlap settings a bit to get good search results
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )
   ```
   Another key difference is how we load a directory full of files. LangChain doesn't have a foolproof way to just load a directory of mixed types. Here we're specificying DOCX through the glob parameter, which selects which files to read, and then the document loader class of DOCX to text loader.
   ```python
   loader = DirectoryLoader(args.docs_dir,
            loader_cls=Docx2txtLoader, # Docx2txtLoader
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.docx"  # which files get loaded
        )
   docs = loader.load()
   ```
   If we wanted to process many different kinds of files, a straightforward approach might be to call directory loader multiple times with different settings, particularly the wildcard passed into the glob parameter and a different document loader. Another approach might be to pre-process the documents into HTML or markdown and work from there.

   This code uses explicit memory for the overall chat using the **conversational buffer memory** class. Each item stored in memory is tagged with an identifier under which it appends accumulated messages. And here we're using "chat history" for that key.
   ```python
   memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
   )
   memory.load_memory_variables({})
   ```
   The actual chat engine, here a **conversational retrieval chain**. This pre-built chain summarizes the query to work better in limited context windows and with longer chat sessions. 
   ```python
   qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
   )
   ```
   There's no single function to call to provide a chat like **REPL** interface, so we roll our own in a while loop. For each conversational turn, both from the human side and from the AI side, we store the message in our memory class.
   ```python
   # Start a REPL loop
   while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input=="exit":
            break
        memory.chat_memory.add_user_message(user_input)
        result = qa_chain({"question": user_input})
        response = result["answer"]
        memory.chat_memory.add_ai_message(response)
        print("AI:", response)
   ```

   ***This doesn't detect any file changes after indexing.*** 

### Document Summarization

One feature that can be paired with RAG is summarization. Write a simple app to take a single or multi-part document in the directory.  We'll use LlamaIndex for this.
1. Fill in the prompt in the startar file along the lines of "Summarize the following document". Tweak this prompt to get the desired granularity and results.

    ```python
    application_prompt = """<insert summarization prompt here>

        DOCUMENT:
    """
    ```
    SOLUTION: 
    ```python
    application_prompt = """Given the following documents,
        summarize them so that each section contains only the most
        important information and relevant facts:

        DOCUMENT:
    """
    ```
2. Write code to process the results of calling *load_data()* on the **SimpleDirectoryReader**. Combine all files into one long string.

    ```python
    fulltext = "<join together docs into one string>"
    ```
    SOLUTION:
    ```python
    fulltext = "\n\n".join([d.get_text() for d in documents])
    ```

> If we need to summarize documents, especially on the input side of a RAG app, we need to think about broader application and data integration ideas.

## Multi-Step AI Workflows with Chaining

So far, we've built small apps that work standalone. But here in the real world, we use AI to create components that exist within a larger app.

If you have table style databases, well then you should use a database and talk to it in SQL. If you need business logic, then you know Java or Rust or Golang or even Python are going to be a far better option than running a full language model. 

So your job as an AI application architect is to define which pieces make sense as conventional application components, and which pieces can use the full firepower of AI. 
In other words, we need to focus on the connections between conventional software and AI software.  And this gets tricky when the fundamental data type of LLMs are unstructured strings.

AI apps, and in particular the adoption of prose as its own programming language require us to think about the interface between components and applications. 
AI orchestration frameworks are at the leading edge of addressing these concerns. Probably, the most common technique involves reliably getting LLM output into JSON for the rest of our app to consume.

---

The first technique we'll talk about is called <mark>**guided prompting**</mark>. If your prompt specifically requests a JSON response, and especially if you include an example, this will usually work, but not with 100 percent consistency across every possible model and configuration. 
If your temperature setting is greater than zero, retrying a prompt will often fix incorrect JSON. And if your app has the flexibility, you can also request YAML. 

**LangChain**

```python
prompt = """Write a weather report for a major city
    in ten words or less.
    Do not include any additional explanation.
"""

guided_prompt = prompt + """
Return the result as JSON as follows:
{ "city": "<CITY_NAME>",
"report": "<BRIEF_REPORT>" }
"""
```

**LlamaIndex**

```python
prompt = ChatMessage(
    role="user",
    content="""Write a weather report for a random city
        in ten words or less.
        Do not include any additional explanation.
""")

guided_prompt = ChatMessage(role="user", content=prompt.content + """
Return the result as JSON as follows:
{ "city": "<CITY_NAME>",
"report": "<SHORT_REPORT>" }
""")
```

Both LangChain and LlamaIndex make use of a third-party library called <mark>**Pydantic**</mark>. It's a useful module in its own right, and it makes it easy to convert back and forth between native Python objects and JSON.  

We're defining a class named after our data type that we're interested in, and it inherits from base model. We put descriptions onto each field here. The LLM will actually use these descriptions to help figure out what goes where

**LangChain**

```python
from langchain.output_parsers import PydanticOutputParser
    from langchain.pydantic_v1 import BaseModel, Field

    class WeatherReport(BaseModel):
        city: str = Field(description="City name")
        report: str = Field(description="Brief weather report")

    parser = PydanticOutputParser(pydantic_object=WeatherReport)
    #print(f"Parser instructions: {parser.get_format_instructions()}")

    runnable_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=parser.get_format_instructions()),
            HumanMessage(content=prompt)
        ]
    )
    chain = runnable_prompt | chat | parser
    py_obj = chain.invoke({})
    print(py_obj.city, py_obj.report)
```

**LlammaIndex**

```python
from pydantic import BaseModel, Field
    from llama_index.program.openai import OpenAIPydanticProgram

    class WeatherReport(BaseModel):
        "A concise weather report for a single city"
        city: str = Field(description="City name")
        report: str = Field(description="Brief weather report")

    program = OpenAIPydanticProgram.from_defaults(
        llm=chat,
        output_cls=WeatherReport,
        prompt_template_str=prompt.content,
        verbose=True,
    )
    print(guided_prompt.content)
    py_obj = program()
    # Now a standard python obj
    print(py_obj.city, py_obj.report)
```
***The guidance prompt that is emitted for the model to follow. Not every model can actually keep up with that.***

There is one other thing you may have heard of called <mark>**JSON mode**</mark>. This is a way from the OpenAI API to ensure that the output gets formatted as JSON. 

But there's a few problems with this. It can enforce that your results are valid JSON with all the curly brackets matching up and so forth, but _it does nothing to make sure the result matches the JSON schema you desire_. For example, the city field could be missing. In this regard, the Pydantic options we just showed, they are just better. 

---

<mark>LLM function calling</mark>

**Task Offloading** : Extending the capabilities of an LLM by delegating tasks or information retrieval.

**Remote Procedure Call** : Extension of the concept of local procedure(function) calls,  across network or other boundaries.

**LlamaIndex**

```python
from llama_index.core.tools import FunctionTool

def get_weather_for_city(city):
    """Get the current weather in a given city"""
    print(f"Calling local get_weather_for_city for {city}")
    return json.dumps({"city": city, "temperature": random.randint(1,50)})

llm = OpenAI(model="gpt-3.5-turbo-1106")
tool = FunctionTool.from_defaults(fn=get_weather_for_city)
agent = OpenAIAgent.from_tools([tool], llm=llm, verbose=True)
response = agent.chat(
    "What's the weather like in Miami?"
)
```
Gives us multiple conversational turns.

**LangChain**

```python
from langchain.chains.structured_output import create_openai_fn_runnable

def get_weather_for_city(city: str):
    """Get the current weather in a given city"""
    print(f"Calling local get_weather_for_city for {city}")
    return json.dumps({"city": city, "temperature": random.randint(1,50)})

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "What's the weather like in {location}?"),
    ]
)

chain = create_openai_fn_runnable([get_weather_for_city], llm, prompt)
response = chain.invoke({"location": "Miami"})
```
Unlike LlamaIndex, the chain that we invoke provides only a single conversational turn. Hence the output is returned as a JSON.

For Multi-turn cases, we need to use LangChain Agents

### Local LLM Task Offloading

The function calling feature we learned about is specific to OpenAI APIs. <mark>Task offloading Functions execute on your local host</mark>

1. Modifythe starter prompt to include a second command, one that matches the get_weather() function.

    ```python
    # task offloading prompt
    system_prompt = """You are a helpful assistant.... in a subsequent conversational turn:

    "#TASK:TIME" to request the current time and date in a format like 2024-01-01T12:34

    Do not include any additional explanation ... without adding any additional explanation.

    Current conversation:
    {history}

    Human: {input}
    AI:"""
    ```

    ```python
    # task offloading prompt
    system_prompt = """You are a helpful assistant. ... in a subsequent conversational turn:

    "#TASK:TIME" to request the current time and date in a format like 2024-01-01T12:34
    "#TASK:WEATHER city" to request the current weather in a given city

    Do not include any additional explanation ... without adding any additional explanation.


    Current conversation:
    {history}

    Human: {input}
    AI:"""
    ```

2. Modify the main chat loop to be able to invoke Python functions based on a request from the LLM.

    ```python
    #Start REPL loop
        while True:
            user_input = input("Ask a question. Type 'exit' to quit.\n>")
            if user_input=="exit":
                break
            result = conversation.invoke({"input": user_input})
            print(result)
            response = result["response"].strip()
            print("AI:", response)
            if response.startswith("#TASK:"):
                print("got a task offloading request...")
                # task offloading
    ```

    ```python
            if response.startswith("#TASK:"):
                cmd = response[6:]
                print(f"got a task offloading request.. {cmd}.")
                
                if cmd=="TIME":
                    observation = get_current_time()
                    print(f"observation: {observation}")
                elif cmd.startswith("WEATHER "):
                    city = cmd[8:]
                    observation = get_weather(city)
                    print(f"observation: {observation}")
                else:
                    print("Unknown task")
                    observation = None
    ```

## Building Agents

### ReAct Agent Framework

**Agent** : In the LLM world, an agent is an interactive chatbot, acting on behalf of a user, that can make use of task offloading.

ReAct Framework is not related to the React JS.

- The 'Act' part is about taking actions or task offloading. 
- The "RE" part is about reasoning, which, through clever prompting, adds a new dimension to the capabilities of LLMs and AI apps. In effect, it lets the AI think about a given problem, decide what needs to happen next, and take action accordingly. 

REACT includes its own internal tracing, divided up into stages, which spell out "the thought process" inside the LLM.

**ReAct Stages**:
1. __Thought__: reasoning about the current situation and what needs to happen next.
2. __Action__: task offloading
3. __Observation__: evaluating results
_The agent can go through multiple cycles of thought, action, observation at its own discretion._

**ReAct Benefits**
- Improved accuracy by reasoning in a coherent and consistent way with high-level planning.
- Reduced hallucination and incorrect facts via feedback loop.
- Improved transparency and interpretability by tracing steps toward a solution.

### Implementing ReAct Agent

- Get an API key at https://www.microsoft.com/en-us/bing/apis/bing-web-search-api

- Put your key in the environment variable **BING_SUBSCRIPTION_KEY**.

### Adding Tools to ReAct Agent

1. Add a math tool to the LangChain agent.

    ```python
    tools = [   
        Tool(
            name="Web Search",
            func=BingSearchAPIWrapper().run,
            description="useful for when you need to answer specific questions from information on the web",
        )
    ]
    ```

    ```python
    from langchain.chains import LLMMathChain

    tools = [   
        Tool(
            name="Web Search",
            func=BingSearchAPIWrapper().run,
            description="useful for when you need to answer specific questions from information on the web",
        ),
        Tool(
            name="Math Calculator",
            func=LLMMathChain.from_llm(llm),
            description="useful for when you need to perform a mathematical calculation",
        ),
    ]
    ```

2. Add a test case to demonstrate math capabilities.

    ```python
    pydict = agent.invoke({"input": "Hi, I am Bob"})
    print(pydict["output"])
    pydict = agent.invoke({"input": "What's my name?"})
    print(pydict["output"])
    pydict = agent.invoke({"input": "Who is the CEO of LinkedIn in 2023?"})
    print(pydict["output"])
    ```

    ```python
    pydict = agent.invoke({"input": "Hi, I am Bob"})
    print(pydict["output"])
    pydict = agent.invoke({"input": "What's my name?"})
    print(pydict["output"])
    pydict = agent.invoke({"input": "Who is the CEO of LinkedIn in 2023?"})
    print(pydict["output"])

    pydict = agent.invoke({"input": "What is 2*pi*10^2"})
    print(pydict["output"])
    ```

> ***Original Repository*** : https://github.com/LinkedInLearning/introduction-to-ai-orchestration-with-langchain-and-llamaindex-3820082/tree/main 
