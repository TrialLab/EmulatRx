from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)

api_key = "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"
client = OpenAI(api_key='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',
                organization='org-p7vlqGOYmscIzHdhbT7MJtEP',
                project='proj_6EKrvoAglwQOkOMLXTRd0SgD')

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

