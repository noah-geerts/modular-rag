from typing import List, Any
from llms.llm import LLM
from openai import OpenAI

class ChatGPT(LLM):
  def __init__(self, openai: OpenAI):
    self.client = openai

  def create_completion(self, prompt: str,
      system_message: str | None = None,
      images_base64: List[str] | None = None) -> str:
    message_content: List[dict[str, Any]] = [{"type": "text", "text": prompt}]
    
    # Append images to message content if we have any
    if images_base64 is not None:
      for image_base64 in images_base64:
        message_content.append({
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    # Create messages and append system prompt if we have one
    messages: List[dict[str, Any]] = [{"role": "user", "content": message_content}]
    if system_message is not None:
      messages.append({"role": "system", "message": system_message})

    # Call openai
    try:
      response = self.client.chat.completions.create(
          model=self.model,
          # ignore typing issues here since it's just an integration and the typing is funny with openai sdk
          messages=messages # type: ignore
      )
      if response.choices[0].message.content is None:
        raise RuntimeError("No content returned from OpenAI API")
      return response.choices[0].message.content
    except:
      raise RuntimeError("Something went wrong while calling the model with the openai SDK in an instance of ChatGPT")