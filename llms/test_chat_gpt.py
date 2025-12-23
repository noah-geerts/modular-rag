import unittest
from unittest.mock import MagicMock
from llms.chat_gpt import ChatGPT

class TestChatGPT(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.chat_gpt = ChatGPT(self.mock_client)
        self.chat_gpt.model = "gpt-4.1"  # Ensure model attribute exists

    def test_create_completion_with_prompt_only(self):
        # Arrange: Mock OpenAI response for prompt-only call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="response text"))]
        self.mock_client.chat.completions.create.return_value = mock_response

        # Act: Call create_completion with just a prompt
        result = self.chat_gpt.create_completion("hello world")

        # Assert: Ensure correct result and correct messages structure
        self.assertEqual(result, "response text")
        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}])

    def test_create_completion_with_system_message(self):
        # Arrange: Mock OpenAI response for system message call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="response text"))]
        self.mock_client.chat.completions.create.return_value = mock_response

        # Act: Call create_completion with a system message
        result = self.chat_gpt.create_completion("hello world", system_message="system prompt")

        # Assert: Ensure correct result and system message included in messages
        self.assertEqual(result, "response text")
        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        self.assertIn({"role": "system", "message": "system prompt"}, kwargs["messages"])

    def test_create_completion_with_images(self):
        # Arrange: Mock OpenAI response for images call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="response text"))]
        self.mock_client.chat.completions.create.return_value = mock_response

        # Act: Call create_completion with images_base64
        images = ["img1base64", "img2base64"]
        result = self.chat_gpt.create_completion("hello world", images_base64=images)

        # Assert: Ensure correct result and images included in messages
        self.assertEqual(result, "response text")
        self.mock_client.chat.completions.create.assert_called_once()
        args, kwargs = self.mock_client.chat.completions.create.call_args
        user_msg = kwargs["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertTrue(any(part["type"] == "image_url" for part in user_msg["content"]))

    def test_create_completion_raises_runtime_error_on_api_error(self):
        # Arrange: Mock OpenAI client to raise an exception
        self.mock_client.chat.completions.create.side_effect = Exception("API error")

        # Act & Assert: Ensure RuntimeError is raised when API call fails
        with self.assertRaises(RuntimeError) as cm:
            self.chat_gpt.create_completion("hello world")
        self.assertIn("Something went wrong", str(cm.exception))

# Edge cases to consider (do not write yet):
# - Response.choices is empty
# - Response.choices[0].message.content is None
# - images_base64 is an empty list
# - system_message is an empty string
# - Non-string prompt or image input