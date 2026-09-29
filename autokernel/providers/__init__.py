"""Model providers behind one interface: complete(messages, tools) -> (reply, usage).

anthropic, openai_compat, mock (canned replies for tests), human (you paste the kernel).
API keys come from the environment only.
"""
