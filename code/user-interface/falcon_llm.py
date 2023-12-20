import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Huggingface Hub Token Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
os.environ['HUGGINGFACEHUB_API_KEY']="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"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Huggingface Hub Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
model_id = 'tiiuae/falcon-7b-instruct'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Template Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
template = """

You are an AI assistant that provides helpful answers to user queries.

{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # LLM Chain Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
falcon_chain=LLMChain(prompt=prompt,
                      llm=falcon_llm,
                      verbose=True)

print(falcon_chain.run("top 5 countries to live in europe"))


question="who is shah rukh khan"
print(falcon_chain.run(question))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Chainlit Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)
    
@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Send the response
    await cl.Message(content=res["text"]).send()