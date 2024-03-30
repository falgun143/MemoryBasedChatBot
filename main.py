from langchain_community.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st


st.title("LangChain Demo using OpenAI API")
input_text=st.text_input("Search the topic you want")


llm = OpenAI( temperature=0.8)

# Memory 
person_memory =ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory =ConversationBufferMemory(input_key='person',memory_key='chat_history')
desc_memory =ConversationBufferMemory(input_key='dob',memory_key='chat_history')

firstinput_prompt_text=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
 )



chain1=LLMChain(llm=llm, prompt=firstinput_prompt_text, verbose=True,output_key='person',memory=person_memory)



secondinput_prompt_text=PromptTemplate(
    input_variables=['person'],
    template="When was the {person} born"
 )

chain2=LLMChain(llm=llm, prompt=secondinput_prompt_text, verbose=True,output_key='dob',memory=dob_memory)


thirdinput_prompt_text=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
 )

chain3=LLMChain(llm=llm, prompt=thirdinput_prompt_text, verbose=True,output_key='description',memory=desc_memory)


parent_chain=SequentialChain(chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander("Person Name"):
        st.info(person_memory.buffer)

    with st.expander("Major Events"):
        st.info(desc_memory.buffer)

