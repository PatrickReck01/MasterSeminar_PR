import ollama

def run_ollama_inference(model_name: str, prompt: str) -> str:
    """
    Runs inference using an Ollama-supported model.

    Args:
        model_name (str): The name of the model, e.g. 'llama3.2:1b', 'deepseek-r1:7b', etc.
        prompt (str): The input text prompt for the model.

    Returns:
        str: The model's response.
    """
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}],options={"temperature": 0.0,"seed": 42})
        
        res_msg =  response['message']['content']

        # For DeepSeek:
        res_msg = extract_summary_only(res_msg)

        print('Response:\n', res_msg)
        #print('--' * 40)

        return res_msg
    
    except Exception as e:
        return f"Error: {str(e)}"



def extract_summary_only(response: str) -> str:
    # Check if there's a </think> tag
    if "</think>" in response:
        # Return everything after the closing tag
        return response.split("</think>")[-1].strip()
    return response.strip()