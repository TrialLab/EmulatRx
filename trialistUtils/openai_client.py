from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)

# Your api_key, organization and project
api_key = ""  
client = OpenAI(api_key='',
                organization='',
                project='')

class OpenaiClient:
    def __init__(self, api_key, organization, project):
        self.client =  OpenAI(api_key=api_key, organization=organization, project=project)

    def get_chatgpt_response(self, prompt, content):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0,
            max_tokens=9999,
            top_p=1
        )

        return response.choices[0].message.content


    def get_chatgpt_response_with_file(self, prompt, filename):
        pdf_assistant = client.beta.assistants.create(
            model="gpt-4o",
            description="An assistant to extract the contents of PDF files.",
            tools=[{"type": "file_search"}],
            name="PDF assistant",
        )

        # Create thread
        thread = client.beta.threads.create()
        file = client.files.create(file=open(filename, "rb"), purpose="assistants")

        # Create assistant
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[
                Attachment(
                    file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
                )
            ],
            content=prompt,
        )

        # Run thread
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
        )

        if run.status != "completed":
            raise Exception("Run failed:", run.status)

        messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
        messages = [message for message in messages_cursor]

        # Output text
        return messages[0].content[0].text.value

