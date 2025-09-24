from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

response = client.chat.completions.create(
  model= 'gpt-3.5-turbo-16k',
  response_format={ "type": "text" },
  messages=[
    # You are a helpful assistant designed to summarize and provide example questions, and then a list of answers to the questions, based on inputs
    {"role": "system", "content": "Given a block of text, summarize it and generate example questions and then answers based on the summary"},
    {"role": "user", "content": """ 

        Lecture 2: Introduction Part 2 - January 9th, 2024
        ARM Cortex-M
        32-bit architecture
        Instruction Set Architecture (ISA)
        Called Thumb ISA and is based on Thumb-2 Technology which supports a mixture of 16-bit and 32 bit technology
        M3 first processor of Cortext Generation
        Released 2005 (M4 was 2010)
        Many Features
        Widely used
        Reduced Instruction Set Computing (RISC)
        IN a typical microcontroller design, the processor takes only a small part of the silicon area.
        Phase Locked Loop
        Enables you to run the microcontroller at different speeds.
        ARM company
        Advanced RISC Machines Ltd.
        Formed 1990
        Introduced ARM6 Processor in 1991
        Purchased in 2016 by SoftBank and attempted to sell to Nvidia but failed.
"""}
  ]
)

print(response.choices[0].message.content)
