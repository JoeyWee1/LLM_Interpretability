import gradio as gr
import subprocess
def reasoning_qwen3(prompt, mode):
    prompt_with_mode = f"{prompt} /{mode}"
    result = subprocess.run(
        ["ollama", "run", "qwen3:8b"],
        input=prompt_with_mode.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()
reasoning_ui = gr.Interface(
    fn=reasoning_qwen3,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Radio(["think", "no_think"], label="Reasoning Mode", value="think")
    ],
    outputs="text",
    title="Qwen3 Reasoning Mode Demo",
    description="Switch between /think and /no_think to control response depth."
)