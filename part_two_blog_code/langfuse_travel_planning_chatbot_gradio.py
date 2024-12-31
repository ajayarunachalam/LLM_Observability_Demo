'''
@author: Ajay
Aim: This is an end-to-end example
      which showcases how to build a Travel Planner AI Assistant chatbot with Ollama, Gradio, and Langfuse for LLM Observability and Evaluation.
'''

import os
import json
import re
import uuid
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display
import gradio as gr
import ollama
from dotenv import load_dotenv
from pydantic import BaseModel
from langfuse import Langfuse
import ast


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

# Flag to check if the initial travel keyword check has been performed
initial_check_done = False


def get_chat_model_completions(messages):
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return ""

def get_chat_model_completions_create(messages):
    try:
        # Convert chat messages to a single string prompt
        if isinstance(messages, list):
            # Extract the system message content
            system_content = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
            prompt = system_content
        else:
            prompt = messages

        response = ollama.generate(
            model="llama3.2:latest",  # llama3.2:latest
            prompt=prompt,
            stream=False
        )
        return response['response']
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return ""

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"
    example_user_req = {'Destination': 'Manali','Package': 'Deluxe','Origin': 'New Delhi','Duration': '4','Budget': '15000 INR'}

    system_message = f"""

    You are an intelligent holiday planner and your goal is to find the best holiday package for a user.
    You need to ask relevant questions and understand the user preferences by analysing the user's responses.
    You final objective is to fill the values for the different keys ('Destination','Package','Origin','Duration','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's preference.
    The python dictionary looks like this {{'Destination': 'values','Package': 'values','Origin': 'values','Duration': 'values','Budget': 'values'}}

    The value for all the keys should be extracted from the user's response.
    The values currently in the dictionary are only representative values.

    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The value for 'Budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 6500 INR. If the user says less than that, please mention that there are no holiday packages in that range.
    - The value for 'Duration' should be a numerical value extracted from the user's response.
    - 'Origin' value can either be 'Mumbai' or 'New Delhi'. If the user mentions any other city, please mention that we are travel company dealing in holidays only from Mumbai and New Delhi
    - For 'Package', give user the option whether they want to go for a 'Premium', 'Deluxe' , 'Luxury' or 'Standard' option.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's preference for the holiday destination and number of nights. \n
    If their primary destination is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys ('Destination','Package','Origin','Duration','Budget') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their requirements. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I want to visit Manali."
    Assistant: "Great! For what duration would you like to visit Manali"
    User: "I would like to visit for 4 nights."
    Assistant: "Thank you for providing that information. Which of the below package would you like to opt for:
    Standard
    Premium
    Deluxe
    Luxury"
    User: "I would like to go with Deluxe"
    Assistant: "Thank you for the information. Could you please let me know whether your origin city would be New Delhi or Mumbai?"
    User: "i'll start from New Delhi"
    Assistant:"Great, thanks. Could you kindly let me know your per person budget for the holiday package? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 15000 inr"
    Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation

def moderation_check(text):
    """Simple moderation check - this is a basic implementation
    You may want to enhance this based on your needs"""
    # List of terms to flag
    flagged_terms = ['hate', 'violence', 'explicit', 'offensive']
    return "Flagged" if any(term in text.lower() for term in flagged_terms) else "Not Flagged"

def extract_dictionary_from_text(text):
    """Extract dictionary values from text using regex patterns."""
    # Pattern to match key-value pairs
    patterns = {
        'Destination': r"'Destination':\s*'([^']*)'",
        'Package': r"'Package':\s*'([^']*)'",
        'Origin': r"'Origin':\s*'([^']*)'",
        'Duration': r"'Duration':\s*'([^']*)'",
        'Budget': r"'Budget':\s*'([^']*)\s*(?:INR)?'"
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = match.group(1).strip()

    # Verify all keys were found
    if len(result) == 5:
        # Clean up the Budget value
        result['Budget'] = result['Budget'].replace(',', '')
        return result
    return None

def dictionary_present(response_input):
    prompt = f"""Return ONLY a dictionary containing the following information from the input.
    DO NOT return any other text, code, or explanations.
    ONLY return the dictionary in this EXACT format:
    {{'Destination': 'CityName', 'Package': 'PackageType', 'Origin': 'CityName', 'Duration': 'Number', 'Budget': 'Number'}}

    Input: {response_input}

    Remember: Return ONLY the dictionary, nothing else."""

    try:
        # Generate response using Ollama
        response = ollama.generate(
            model='llama3.2:latest',
            prompt=prompt,
            stream=False
        )

        # Extract dictionary from the response
        result = extract_dictionary_from_string(response_input)
        if result:
            return result

        # If direct extraction failed, try to extract from the model's response
        result = extract_dictionary_from_string(response['response'])
        if result:
            return result

        raise Exception("Could not extract dictionary from response")

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'Destination','Package','Origin','Duration','Budget'
    Next you need to evaluate if the keys have the the values filled correctly.
    - The value for the key 'Budget' needs to contain a number with currency.
    - The value of key 'Package' needs to be either  'Premium', 'Deluxe' , 'Luxury' or 'Standard'
    - The value of key 'Duration' needs to contain a number
    - The value of key 'Origin' should either be 'New Delhi' or 'Mumbai'
    Output a string 'Yes' if the input contains the dictionary with the values correctly filled for all keys.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        # Make the API call with properly formatted messages
        confirmation = ollama.chat(
            model="llama3.2:latest",
            messages=messages
        )

        # Extract the content from the response
        confirmation_response = confirmation['message']['content']

        # Optional: You might want to add some validation here to ensure
        # the response is actually a dictionary string
        return confirmation_response

    except Exception as e:
        print(f"Error in intent extraction: {e}")
        return " "  # Return empty string in case of error

def extract_duration(itinerary):
    """Extract total duration from itinerary string."""
    numbers = re.findall(r'(\d+)N', itinerary)
    return str(sum(int(n) for n in numbers))

def extract_destination(dest_str):
    """Extract first destination before '|' or return full string."""
    return dest_str.split('|')[0].strip()

def product_map_layer(row):
    """Product mapping"""
    try:
        return {
            'Destination': extract_destination(row['Destination']),
            'Package': row['Package Type'].strip(),
            'Origin': row['Start City'].strip(),
            'Duration': extract_duration(row['Itinerary'])
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def extract_dictionary_from_string(text):
    """Extract dictionary from string using regex."""
    if isinstance(text, dict):
        return text

    patterns = {
        'Destination': r"'Destination':\s*'([^']*)'",
        'Package': r"'Package':\s*'([^']*)'",
        'Origin': r"'Origin':\s*'([^']*)'",
        'Duration': r"'Duration':\s*'([^']*)'",
        'Budget': r"'Budget':\s*'([^']*)\s*(?:INR)?'" #
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, str(text))
        if match:
            result[key] = match.group(1).strip()

    return result if result else None

def compare_holiday_with_user(user_req_string):
    """Holiday comparison function."""
    try:
        # Load data
        holiday_df = pd.read_csv('makemytrip_dataset.csv')
        display(holiday_df.head(2))

        # Extract user requirements
        user_requirements = extract_dictionary_from_string(user_req_string)#extract_dictionary_from_string
        if not user_requirements:
            raise ValueError("Could not parse user requirements")

        # Process budget
        budget = int(user_requirements.get('Budget', '0').replace(',', '').split()[0])
        del user_requirements['Budget']

        # Filter by budget first (this reduces the dataset for subsequent operations)
        filtered_holiday = holiday_df[holiday_df['Per Person Price'] <= budget].copy()

        # Apply product mapping in parallel
        with ThreadPoolExecutor() as executor:
            holiday_features = list(executor.map(
                product_map_layer,
                [row for _, row in filtered_holiday.iterrows()]
            ))

        filtered_holiday['holiday_feature'] = holiday_features

        # Remove rows where feature extraction failed
        filtered_holiday = filtered_holiday.dropna(subset=['holiday_feature'])

        # Efficient comparison using vectorized operations
        def match_requirements(feature_dict):
            try:
                return all(
                    str(feature_dict.get(key, '')).lower() == str(value).lower()
                    for key, value in user_requirements.items()
                )
            except:
                return False

        mask = filtered_holiday['holiday_feature'].apply(match_requirements)
        result_df = filtered_holiday[mask]

        # Select only necessary columns and convert to JSON
        #result_columns = ['Package Name','Destination', 'Package Type', 'Start City', 'Itinerary', 'Per Person Price','Sightseeing Places Covered']
        #return result_df[result_columns].to_json(orient='records')
        return result_df.to_json(orient='records')

    except Exception as e:
        print(f"Error in compare_holiday_with_user: {e}")
        return "[]"

def format_holiday_summary(products):
    """Format holiday products into a readable summary."""
    try:
        # Parse JSON string to list of dictionaries if it's a string
        if isinstance(products, str):
            products = json.loads(products)

        # Sort products by price in descending order
        sorted_products = sorted(products, key=lambda x: float(x.get('Per Person Price', 0)), reverse=True)

        summary = []
        for i, product in enumerate(sorted_products, 1):
            # Extract major attractions
            destinations = product['Destination'].split('|')
            attractions = ', '.join(destinations)

            # Format the summary line
            summary_line = (
                f"{i}. {product['Package Name']}: "
                f"Visit {attractions}, "
                f"₹{int(product['Per Person Price']):,} per person"
            )
            summary.append(summary_line)

        return '\n'.join(summary)
    except Exception as e:
        print(f"Error formatting holiday summary: {e}")
        return str(products)

def initialize_conv_reco(products):
    try:
        # Format the holiday summaries
        holiday_summary = format_holiday_summary(products)
        system_message = f"""
        You are an intelligent holiday expert and you are tasked with the objective to \
        solve the user queries about any product from the catalogue: {products}.\
        You should keep the user requirements in mind while answering the questions.\
        {holiday_summary}

        Start with a brief summary of each holiday in the following format, in decreasing order of price per person:
        1. <Holiday Name> : <Major attractions of the holiday>, <Price per person in Rs>
        2. <Holiday Name> : <Major attractions of the holiday>, <Price per person in Rs>

        Provide a precise summary:

        """
        conversation = [{"role": "system", "content": system_message}]
        return conversation
    except Exception as e:
      print(f"Error initializing conversation: {e}")
      return ""

# Add decorator here to capture overall timings, input/output, and manipulate trace metadata via `langfuse_context`
@observe()
async def create_response(prompt: str, history):
    global initial_check_done
    # Save trace id in global var to add feedback later
    global current_trace_id
    current_trace_id = langfuse_context.get_current_trace_id()
    print(f"Current trace ID: {current_trace_id}")

    # Add session_id to Langfuse Trace to enable session tracking
    global session_id
    langfuse_context.update_current_trace(
        name="travel_company_ai_assistant",
        session_id=session_id,
        input=prompt,
    )

    # Add prompt to history
    if not history:
        history = initialize_conversation()
    history.append({"role": "user", "content": prompt})
    yield history

    try:
        # Get completion via ollama
        response = {"role": "assistant", "content": ""}
        ollama_response = ollama.chat(
            messages=history,
            model="llama3.2:latest",
        )
        response["content"] = ollama_response.message.content or ""

        # Customize trace output for better readability in Langfuse Sessions
        langfuse_context.update_current_trace(
            output=response["content"],
        )

        yield history + [response]
    except Exception as e:
        print(f"Error generating response: {e}")
        response = {"role": "assistant", "content": "I'm sorry, I encountered an error. Please try again."}
        langfuse_context.update_current_trace(
            output=response["content"],
        )
        yield history + [response]

async def respond(prompt: str, history):
    global initial_check_done
    # Filter non-travel-related queries only for the initial conversation
    if not initial_check_done:
        travel_keywords = ["travel", "trip", "flight", "hotel", "vacation", "booking", "destination", "itinerary"]
        if not any(keyword in prompt.lower() for keyword in travel_keywords):
            response = {"role": "assistant", "content": "I'm sorry, I can only assist with travel-related queries."}
            yield history + [response]
            return
        initial_check_done = True

    async for message in create_response(prompt, history):
        yield message

# Add decorator here to capture overall timings, input/output, and manipulate trace metadata via `langfuse_context`
@observe()
def chat(message, history):
    """
    Handle chat messages and maintain conversation state
    """
    global initial_check_done
    # Save trace id in global var to add feedback later
    global current_trace_id
    current_trace_id = langfuse_context.get_current_trace_id()
    print(f"Current trace ID: {current_trace_id}")

    # Add session_id to Langfuse Trace to enable session tracking
    global session_id
    langfuse_context.update_current_trace(
        name="travel_company_ai_assistant",
        session_id=session_id,
        input=message,
    )
    # Initialize or continue conversation
    if not history:
        conversation = initialize_conversation()
    else:
        conversation = [{"role": "system", "content": initialize_conversation()[0]["content"]}]
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
    
    # Add new message and get response
    conversation.append({"role": "user", "content": message})
    response = get_chat_model_completions(conversation)

    response_dict_n = dictionary_present(response)
    print(response_dict_n)
    
    # Check if we have all the information and can make recommendations
    #if "'Destination'" in response and "'Budget'" in response:
    # dict_result = extract_dictionary_from_string(response_dict_n)
    # print(dict_result)
    if response_dict_n:
        recommended_holiday = compare_holiday_with_user(response_dict_n)
        print(recommended_holiday)
        if recommended_holiday != "[]":
            holiday_summary = format_holiday_summary(recommended_holiday)
            response += f"\n\nBased on your preferences, here are the recommended holiday packages:\n\n{holiday_summary}"
        else:
            response += "\n\nI apologize, but I couldn't find any holiday packages matching your exact criteria. Would you like to adjust any of your preferences?"

    return response

def handle_like(data: gr.LikeData):
    print(f"handle_like called with data: {data}")  # Debugging statement
    global current_trace_id
    print(f"Handling like: {data.liked}, Trace ID: {current_trace_id}")
    try:
        if data.liked:
            print(f"Sending positive feedback to Langfuse with trace ID: {current_trace_id}")
            langfuse.score(value=1, name="user-feedback", trace_id=current_trace_id)
        else:
            print(f"Sending negative feedback to Langfuse with trace ID: {current_trace_id}")
            langfuse.score(value=0, name="user-feedback", trace_id=current_trace_id)
    except Exception as e:
        print(f"Error sending feedback to Langfuse: {e}")

def handle_comment(comment: str):
    global current_trace_id
    print(f"Handling comment: {comment}, Trace ID: {current_trace_id}")
    if comment:
        try:
            print(f"Sending comment to Langfuse with trace ID: {current_trace_id}")
            trace_id = langfuse.trace(
                name="user-comment",
                metadata={"comment": comment},
                session_id=session_id
            )
            print(f"New trace ID for comment: {trace_id}")
        except Exception as e:
            print(f"Error sending comment to Langfuse: {e}")
    else:
        print(f"Skipping comment for trace ID: {current_trace_id}")

def handle_survey(satisfaction: int, additional_comments: str):
    global current_trace_id
    print(f"Handling survey: Satisfaction: {satisfaction}, Comments: {additional_comments}, Trace ID: {current_trace_id}")
    try:
        trace_id = langfuse.trace(
            name="user-survey",
            metadata={"satisfaction": satisfaction, "additional_comments": additional_comments},
            session_id=session_id
        )
        print(f"New trace ID for survey: {trace_id}")
    except Exception as e:
        print(f"Error sending survey to Langfuse: {e}")

async def handle_retry(history, retry_data: gr.RetryData):
    new_history = history[: retry_data.index]
    previous_prompt = history[retry_data.index]["content"]
    async for message in respond(previous_prompt, new_history):
        yield message

# Create Gradio interface 
with gr.Blocks() as demo:
    gr.Markdown("# Travel Assist AI Planner Chatbot using Ollama + Gradio + Langfuse")
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, "https://static.langfuse.com/cookbooks/gradio/hf-logo.png"),
        height=600
    )
    
    msg = gr.Textbox(
        show_label=False,
        placeholder="Tell me about your travel plans...",
        container=False
    )
    
    clear = gr.ClearButton([msg, chatbot])

    # Correct way to handle message submission
    def respond(message, chat_history):
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    chatbot.retry(handle_retry, chatbot, [chatbot])
    chatbot.like(handle_like, None, None)
    chatbot.clear(set_new_session_id)

    comment_box = gr.Textbox(label="Do you want to add a comment?", placeholder="Enter your comment here...")
    submit_comment_button = gr.Button("Submit Comment")
    skip_comment_button = gr.Button("Skip")

    submit_comment_button.click(handle_comment, comment_box, None)
    skip_comment_button.click(handle_comment, "", None)

    survey_satisfaction = gr.Slider(1, 5, step=1, label="How satisfied are you with the chatbot's response? (1-5)")
    survey_comments = gr.Textbox(label="Additional Comments", placeholder="Enter any additional comments here...")
    submit_survey_button = gr.Button("Submit Survey")

    submit_survey_button.click(handle_survey, [survey_satisfaction, survey_comments], None)

if __name__ == "__main__":
    demo.launch(share=True)