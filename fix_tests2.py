import glob, re

for f in glob.glob('tests/*.py'):
    c = open(f, encoding='utf-8').read()
    
    # 1. Remove the old generate
    c = re.sub(
        r'def generate\(self, messages: Sequence\[ChatMessage\]\) -> ChatResponse:(?:\s*# pragma: no cover)?\n\s*raise AssertionError\("ScriptedProvider\.generate should not be used in these tests\."\)\n', 
        '', 
        c
    )
    
    # 2. Rename generate_with_tools to generate, and unpack request
    c = re.sub(
        r'def generate_with_tools\(\s*self,\s*messages: list\[dict\[str, object\]\],\s*tools: list\[dict\[str, object\]\],\s*\) -> LLMResponse:',
        'def generate(self, request, model_id="test") -> LLMResponse:\n        messages = request.messages\n        tools = request.tools',
        c
    )
    c = re.sub(
        r'def generate_with_tools\(\s*self,\s*messages: list\[dict\[str, Any\]\],\s*tools: list\[dict\[str, Any\]\],\s*\) -> LLMResponse:',
        'def generate(self, request, model_id="test") -> LLMResponse:\n        messages = request.messages\n        tools = request.tools',
        c
    )
    
    # 3. Add provider_id
    c = c.replace('class ScriptedProvider(LLMProvider):', 'class ScriptedProvider(LLMProvider):\n    @property\n    def provider_id(self) -> str: return "scripted"\n')
    
    # 4. Fix tool_call= -> tool_calls=
    c = re.sub(r'tool_call=(ToolCall\(.*?\))', r'tool_calls=[\1]', c)

    open(f, 'w', encoding='utf-8').write(c)

