#! /usr/bin/env python
'''
@author: Ajay
Aim: This is a simple end-to-end example
      which showcases how to build a travel domain-specific chatbot with Ollama, Gradio, and Langfuse for LLM Observability and Evaluation.
'''
import ollama
import uuid
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

# Add decorator here to capture overall timings, input/output, and manipulate trace metadata via `langfuse_context`
@observe()
async def create_response(prompt: str, history):
    # Save trace id in global var to add feedback later
    global current_trace_id
    current_trace_id = langfuse_context.get_current_trace_id()
    print(f"Current trace ID: {current_trace_id}")

    # Add session_id to Langfuse Trace to enable session tracking
    global session_id
    langfuse_context.update_current_trace(
        name="travel_bot",
        session_id=session_id,
        input=prompt,
    )

    # Add prompt to history
    if not history:
        history = [{"role": "system", "content": "You are a travel company AI assistant. Only respond to travel-related queries."}]
    history.append({"role": "user", "content": prompt})
    yield history

    try:
        # Get completion via ollama
        # Auto-instrumented by Langfuse via the import, see alternative in note above
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
    # Filter non-travel-related queries
    travel_keywords = ["travel", "visit",  "trip", "flight", "hotel", "vacation", "booking", "destination", "itinerary", "plan","holiday"]
    if not any(keyword in prompt.lower() for keyword in travel_keywords):
        response = {"role": "assistant", "content": "I'm sorry, I can only assist with travel-related queries."}
        yield history + [response]
        return

    async for message in create_response(prompt, history):
        yield message

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

with gr.Blocks() as demo:
    gr.Markdown("# Travel Chatbot using Ollama + Gradio + Langfuse")
    chatbot = gr.Chatbot(
        label="Chat",
        type="messages",
        show_copy_button=True,
        avatar_images=(
            None,
            "https://static.langfuse.com/cookbooks/gradio/hf-logo.png",
        ),
    )
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    prompt.submit(respond, [prompt, chatbot], [chatbot])
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
    demo.launch(share=True, debug=True)
