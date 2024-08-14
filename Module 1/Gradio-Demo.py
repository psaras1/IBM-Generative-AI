import gradio as gr

# Form accepts user input for name and intensity(through slider)
def greet(name,intensity):
  return "Hello, " + name + "!" * int(intensity)
# The Interface class is designed to create demos for machine learning models
# that accept one or more inputs and return one or more outputs
# (It has 3 core components: Interface(fn), Inputs, and Outputs)
demo = gr.Interface(
    # Function to wrap the UI around
    fn=greet,
    # Number of components should match the function's arguments
    # The inputs and outputs components can take one or more Gradio components
    inputs=["text", "slider"],
    outputs="text"
)
# Launch the web interface
demo.launch()