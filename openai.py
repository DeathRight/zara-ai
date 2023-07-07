import json
import os
import openai
from tiktoken import Tokenizer
from zara_personality import primer as zara_primer

openai.api_key = os.getenv("openai")


class OpenAIChatbot:
    """
    A helper class for interacting with the OpenAI completion API.

    After every message and response, the conversation history is trimmed to ensure that it does not exceed the maximum token size and
    leaves room for the assistant's response.

    Instance variables:
        model: The name of the model to use.
        initial_messages: The list of initial messages used to prime the chatbot - will always be the first messages in the conversation history.
        conversation_history: The list of messages in the conversation history.
        max_token_size: The maximum number of tokens allowed in the conversation history.
        trim_message_callback: A callback function that is called when a message (or messages) is trimmed from the conversation history.

    """

    def __init__(
        self,
        personality_primer: list[dict[str, str]] = zara_primer,
        use_large_model=False,
        initial_messages: list[dict[str, str]] = None,
        messages: list[dict[str, str]] = None,
        functions: list[dict[str, str]] = None,
        functions_callable: list[dict[str, function]] = None,
        max_token_size=None,
        trim_message_callback=None,
        error_callback=None,
        response_token_size=None,
    ):
        """
        Creates a new OpenAIChatbot instance.

        Args:
            personality_primer: A list of messages to use as the personality primer, in the same format as OpenAI completion API messages. ([{"role": ..., "content": ...}, ...}])
            use_large_model: Whether to use the large model (gpt-3.5-turbo-16k) or the small model (gpt-3.5-turbo) - defaults to False.
            initial_messages: A list of messages to use as the initial messages along with the personality primer - defaults to None.
            messages: A list of messages to start the conversation with - defaults to None.
            functions: A list of function descriptions to be passed to the chatbot - defaults to None.
            functions_callable: A list of actual functions corresponding to the function descriptions (Must use the same names as in 'functions') - defaults to None.
            max_token_size: The maximum number of tokens allowed in the conversation history - defaults to 16000 if use_large_model is True, otherwise 4000.
            trim_message_callback: A callback function that is called when a message (or messages) is trimmed from the conversation history - defaults to None.
            response_token_size: The number of tokens to use for the assistant's response - defaults to the API's default value.
        """

        # Settings
        self.model = "gpt-3.5-turbo-16k" if use_large_model else "gpt-3.5-turbo"
        self.max_token_size = max_token_size or (16000 if use_large_model else 4000)
        self.trim_message_callback = trim_message_callback
        self.response_token_size = response_token_size

        # Functions
        self.functions = functions
        self.functions_callable = functions_callable

        # Personality primer
        self.initial_messages = personality_primer

        if initial_messages:
            self.initial_messages.extend(initial_messages)

        # Conversation history
        self.conversation_history = self.initial_messages.copy()

        # Add messages to the conversation history
        if messages:
            self.conversation_history.extend(messages)

    def create_response(self):
        """
        Wrapper function for the OpenAI completion API to avoid boilerplate code.
        """
        return openai.ChatCompletion.create(
            model=self.model,
            messages=self.conversation_history,
            functions=self.functions,
            max_tokens=self.response_token_size,
        )

    def count_tokens(self, messages: list):
        """
        Counts the total number of tokens in a list of messages.

        Args:
            messages: A list of messages.

        Returns:
            The total number of tokens in the list of messages.
        """
        tokenizer = Tokenizer()
        return sum(tokenizer.count_tokens(message["content"]) for message in messages)

    def update_conversation_history(self, message_content: str, role="user", name=None):
        """
        Adds a message to the conversation history and trims the history if necessary.

        Args:
            message_content: The content of the message to add
            role: The role of the message to add - defaults to "user"

        Returns:
            The number of tokens available in the conversation history after the update.
        """
        if message_content is not None:
            # Check if function, if so, add name
            if role == "function":
                new_message = {"role": role, "content": message_content, "name": name}
            else:
                new_message = {"role": role, "content": message_content}

            self.conversation_history.append(new_message)

        tokenizer = Tokenizer()

        # Check the total tokens used so far
        total_tokens = self.count_tokens(self.conversation_history)
        available_tokens = self.max_token_size - total_tokens

        callback_invoked = False

        # When we have more than one user message in the history (excluding the initial messages)
        if len(self.conversation_history) > len(self.initial_messages) + 1:
            while available_tokens < self.response_token_size:
                # We remove the oldest user message after the initial messages
                if len(self.conversation_history) > len(self.initial_messages) + 1:
                    self.conversation_history.pop(
                        len(self.initial_messages) + 1
                    )  # Remove the oldest conversation message after initial messages
                    total_tokens = self.count_tokens(self.conversation_history)
                    available_tokens = self.max_token_size - total_tokens

                    # If the callback hasn't been invoked yet, invoke it
                    if not callback_invoked and self.trim_message_callback:
                        callback_invoked = True

                else:
                    break

            # After trimming, call the callback
            if self.trim_message_callback and callback_invoked:
                self.trim_message_callback()
                # TODO: track how many messages/characters were trimmed and provide that info to the callback

        # If there's only one user message and it's too long
        elif (
            len(self.conversation_history) == len(self.initial_messages) + 1
            and available_tokens < self.response_token_size
        ):
            user_message = self.conversation_history[-1]["content"]
            text_tokens = list(tokenizer.tokenize(user_message))
            available_tokens_for_message = available_tokens - self.response_token_size

            if len(text_tokens) > available_tokens_for_message:
                # Trim the message
                user_message = "".join(
                    [
                        token.string
                        for token in text_tokens[:available_tokens_for_message]
                    ]
                )
                self.conversation_history[-1][
                    "content"
                ] = user_message  # Replace the message with the trimmed message
                available_tokens = (
                    self.response_token_size
                )  # Now we have just enough tokens for the assistant's response

                if self.trim_message_callback:
                    self.trim_message_callback()

        return available_tokens

    def generate_response(self):
        """
        Generates a response using the OpenAI completion API. If the response does not end with punctuation, it will try to generate a continuation.

        Returns:
            The response from the OpenAI completion API.
        """
        response = self.create_response()

        response_message = response.choices[0].message

        # Check if the AI wanted to call a function
        if response_message.get("function_call"):
            func = response_message.get("function_call")
            func_name = func.get("name")
            func_args = json.loads(func.get("arguments"))

            # Check if the function exists
            if func_name in self.functions_callable:
                # Try to call the function, handle exceptions
                try:
                    # Call the function with unpacked arguments
                    func_result = self.functions_callable[func_name](**func_args)

                    # Add the function result to the conversation history
                    self.update_conversation_history(
                        message_content=func_result, name=func_name, role="function"
                    )

                    # Generate second response where the AI can respond to the function result
                    # and restart the process (check if function, check if ends in punctuation, etc.)
                    #
                    # We will continue to do this until the AI gives an assistant response that ends in punctuation.
                    return self.generate_response()

                except Exception as e:
                    # Call the error callback with information
                    if self.error_callback:
                        self.error_callback(func_name, str(e))

                    # Remove the last message where the AI tried to call the function
                    self.rollback()

                    # Ask the AI to apologize and explain the error
                    return self.respond(
                        f"An error occurred when trying to execute the function '{func_name}'. Please tell the user and apologize. (Don't ask to troubleshoot)"
                    )

            else:
                print(f"Function {func_name} is not recognized.")

        reply = response.choices[0].message.get("content")

        # If the response does not end with punctuation, we give it one more shot to finish
        if reply[-1] not in {".", "!", "?"}:
            available_tokens = self.update_conversation_history(reply, "assistant")

            if available_tokens > 0:
                continuation_response = self.create_response()

                continuation_reply = continuation_response.choices[0].message.get(
                    "content"
                )
                reply += continuation_reply

        return reply

    def rollback(self):
        """
        Removes the last message from the conversation history.
        """
        self.conversation_history.pop()

    def add_messages(self, messages: list):
        """
        Adds a list of messages to the conversation history.
        """
        for message in messages:
            self.update_conversation_history(message)

    def respond(self, user_message):
        """
        Adds a message to the conversation history and generates a response using the OpenAI completion API.

        Tries to keep running until the AI provides an assistant response that ends in punctuation,
        allowing the AI to call functions consecutively.

        Args:
            user_message: The message to add to the conversation history.

        Returns:
            The response from the OpenAI completion API.
        """
        self.update_conversation_history(user_message)
        return self.generate_response()

    # Get conversation history, minus the initial messages
    def get_conversation(self):
        return self.conversation_history[len(self.initial_messages) :]
