import tiktoken

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    Counts the number of tokens in a given text using tiktoken as a fast proxy.
    """
    try:
        encoding = tiktoken.get_encoding(model_name)
    except ValueError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
