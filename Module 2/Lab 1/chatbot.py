# AutoModelForSeq2SeqLM: Allows interaction with chosen language model
# AutoTokenizer: Optimizes input and passes it to the language model efficiently
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
while True:
    # The transformers library function expects to receive the conversation history as a string, 
    # with each element separated by the newline character '\n'
    history_string = "\n".join(conversation_history)

    input_text = input("> ")

    # return_tensors="pt" instructs function to return output as PyTorch tensors
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # ** operator is used to unpack the inputs dictionary 
    outputs = model.generate(**inputs)

    # Decode the output(tensor to string)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    conversation_history.append(input_text)
    conversation_history.append(response)
