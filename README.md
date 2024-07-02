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

        

