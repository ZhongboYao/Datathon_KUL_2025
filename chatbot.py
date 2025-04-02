import api
from openai import OpenAI

def respond(query, context):
    openai_client = OpenAI(api_key=api.OPENAI_API)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who will answer the user's question based only on (but not only on) the relevant parts "
                "of the provided context. Do not bring in information that is not directly related to the user's question. "
                "Provide a concise explanation of the question and how it relates to the relevant context. "
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's question is: {query}\n\n"
                "Below is some background information:\n\n"
                f"{context}\n\n"
            )
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
    )

    return response.choices[0].message.content

def answer_with_knowledge(query, context, knowledge):
    openai_client = OpenAI(api_key=api.OPENAI_API)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who will answer the user's question based only on (but not only on) the relevant parts "
                "of the provided context about climate policies and provided climate knowledges. Do not bring in information that is not directly related to the user's question. "
                "Provide a concise explanation of the question and how it relates to the relevant context. "
            )
        },
        {
            "role": "user",
            "content": (
                f"The user's question is: {query}\n\n"
                "Below is the climate policy context:\n\n"
                f"{context}\n\n"
                "Below is the knowledges that may be useful:\n\n"
                f"{knowledge}\n\n"
            )
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1000,
    )

    return response.choices[0].message.content