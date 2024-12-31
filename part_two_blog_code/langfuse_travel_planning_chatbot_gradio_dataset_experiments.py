#! /usr/bin/env python
'''
@author: Ajay
Aim: This is a simple end-to-end example
      which showcases how to build Dataset & run Experiments in Langfuse for LLM Observability and Evaluation.
'''
import ollama
import uuid
import pandas as pd
import re
import ast
from dotenv import load_dotenv
from pydantic import BaseModel
from langfuse import Langfuse
import gradio as gr
import json

# Load environment variables
load_dotenv()

langfuse = Langfuse()

session_id = None

def set_new_session_id():
    global session_id
    session_id = str(uuid.uuid4())
    print(f"New session ID: {session_id}")

# Initialize
set_new_session_id()

# Langfuse decorator
from langfuse.decorators import observe, langfuse_context

# Global reference for the current trace_id which is used to later add user feedback
current_trace_id = None

system_prompt = "You are a travel company AI assistant. Only respond to travel-related queries."

local_items = [
    {
        "input": {
            "Destination": "Manali",
            "Package": "Deluxe",
            "Origin": "New Delhi", 
            "Duration": "4",
            "Budget": "15000 INR"
        },
        "expected_output": {
            "Holiday Name": "Experiential Manali from Chandigarh (Candid Photography)",
            "Major attractions": "Vashishth Kund | Hadimba Temple | Tibetan Monastery | Personal Photoshoot in Manali | Solang Valley",
            "Price": "6023"
        }
    },
    {
        "input": {
            "Destination": "Jaipur",
            "Package": "Luxury",
            "Origin": "New Delhi",
            "Duration": "2", 
            "Budget": "12000 INR"
        },
        "expected_output": {
            "Holiday Name": "Exotic Jaipur",
            "Major attractions": "City Palace | Hawa Mahal | Jantar Mantar | Amer Fort | Mehrangarh Fort",
            "Price": "10023"
        }
    }
]

for item in local_items:
    langfuse.create_dataset_item(
        dataset_name="trip_planning_dataset",
        # input=item["input"],
        # expected_output=item["expected_output"]
        input=json.dumps(item["input"]),  # Convert input dictionary to JSON string
        expected_output=json.dumps(item["expected_output"])  # Convert expected_output dictionary to JSON string
    )

    

# @observe()
# def run_my_custom_llm_app(input_data, system_prompt):
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": input_data["input"]}
#     ]
#     completion = ollama.chat(model="llama3.2:latest", messages=messages).message.content
#     return completion

@observe()
def run_my_custom_llm_app(input_data, system_prompt):
    if isinstance(input_data, str):
        input_dict = json.loads(input_data)
    else:
        input_dict = input_data

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(input_dict)}  # Ensure the input is correctly formatted
    ]
    completion = ollama.chat(model="llama3.2:latest", messages=messages).message.content
    return completion


# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
# you can also use LLM-as-a-judge managed within Langfuse to evaluate the outputs
def simple_evaluation(output, expected_output):
  return output == expected_output


def run_experiment(experiment_name, system_prompt):
  dataset = langfuse.get_dataset("trip_planning_dataset")
 
  for item in dataset.items:
    # item.observe() returns a trace_id that can be used to add custom evaluations later
    # it also automatically links the trace to the experiment run
    with item.observe(run_name=experiment_name) as trace_id:
 
      # run application, pass input and system prompt
      output = run_my_custom_llm_app(item.input, system_prompt)
 
      # optional: add custom evaluation results to the experiment trace
      # we use the previously created example evaluation function
      langfuse.score(
        trace_id=trace_id,
        name="exact_match",
        value=simple_evaluation(output, item.expected_output)
      )

from langfuse.decorators import langfuse_context
 
# run_experiment(
#     "directly_ask",
#     "The user will input their travel requirements that includes 'Destination', 'Package','Origin', 'Duration', and 'Budget', respond with the travel iternary that includes Holiday Package Name, Attractions Covered, and Price per person."
# )
# run_experiment(
#     "asking_specifically",
#     "Hi, I want to travel to a destination"
# )
# run_experiment(
#     "asking_specifically_1st_try",
#     "The user will input where they want to travel"
# )
# run_experiment(
#     "asking_specifically_2nd_try",
#     "The user will input their travel requirements"
# )

run_experiment(
    "asking_specifically_3rd_try",
    "You are a travel company AI assistant. Only respond to travel-related queries. The user will input their travel requirements"
)
 
# Assert that all events were sent to the Langfuse API
langfuse_context.flush()
langfuse.flush()