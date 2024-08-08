import os
import re


def create_llm(llm_type: str, model: str, verbose: bool = False, callbacks=None, **kwargs):
    if llm_type == "openai":
        import langchain_openai

        return langchain_openai.ChatOpenAI(model_name=model, verbose=verbose)

    if llm_type == "azure-openai":
        import langchain_openai

        return langchain_openai.AzureChatOpenAI(
            azure_deployment=model,
            openai_api_version=os.environ.get("OPENAI_API_VERSION"),
            callbacks=callbacks,
            verbose=verbose,
            model_kwargs=kwargs.get("model_kwargs", {}),
        )

    if llm_type == "vertexai":
        import langchain_google_vertexai as google_ai

        return google_ai.ChatVertexAI(model_name=model, callbacks=callbacks, verbose=verbose)

    if llm_type == "aws-redrock":
        # from langchain_community.chat_models import BedrockChat
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            credentials_profile_name=os.environ.get("AWS_PROFILE"),
            model_id=model,
            callbacks=callbacks,
            verbose=verbose,
            model_kwargs={"temperature": kwargs.get("temperature", 0)},
        )

    if llm_type == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, callbacks=callbacks, verbose=verbose, temperature=kwargs.get("temperature", 0)
        )


def remove_thinking(text: str) -> str:
    # For anthropic
    pattern = r"<thinking>(.*?)</thinking>\s*"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    return text


def remove_markdown(text: str) -> str:
    pattern = r".*```(?:json|python)\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1)

    return text


def get_tag_code(text: str) -> str:
    pattern = r"<CODE>(.*?)</CODE>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    return ""


def get_tag_answer(text: str) -> str:
    pattern = r"<ANSWER>(.*?)</ANSWER>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    return "I don't know"


def get_tag_data(text: str) -> str:
    pattern = r"<DATA>(.*?)</DATA>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    return ""
