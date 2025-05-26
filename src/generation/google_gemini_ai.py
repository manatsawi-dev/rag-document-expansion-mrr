import google.generativeai as genai
import src.constants.constants as constants


genai.configure(api_key=constants.GOOGLE_GEMINI_API_KEY)
gemini_client = genai.GenerativeModel(constants.GOOGLE_GEMINI_MODEL_NAME)


def gemini_generate_content(
    prompt: str,
    max_output_tokens: int = 8192,
    response_schema=None,
) -> any:
    if not constants.GOOGLE_GEMINI_API_KEY:
        raise ValueError(
            "Google Gemini API key is not set. Please set it in constants.py."
        )
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    generation_config = {
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
        "response_schema": response_schema,
    }

    response = gemini_client.generate_content(
        prompt,
        generation_config=generation_config,
    )
    return response.text
