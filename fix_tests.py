import glob

def patch_file(p):
    with open(p, encoding='utf-8') as f:
        c = f.read()
    
    if 'def generate(' not in c and 'def generate_with_tools(' not in c:
        return
        
    c = c.replace('def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:  # pragma: no cover\n        raise AssertionError("ScriptedProvider.generate should not be used in these tests.")', '')
    
    c = c.replace('def generate(self, messages: Sequence[ChatMessage]) -> ChatResponse:\n        raise AssertionError("ScriptedProvider.generate should not be used in these tests.")', '')

    c = c.replace('def generate_with_tools(', 'def generate(self, request, model_id="test"): \n        messages = request.messages \n        tools = request.tools \n        def _generate_with_tools(')
    
    c = c.replace('class ScriptedProvider(LLMProvider):', 'class ScriptedProvider(LLMProvider):\n    @property\n    def provider_id(self) -> str: return "scripted"\n')
    
    c = c.replace('tool_call=ToolCall', 'tool_calls=[ToolCall')
    # we need to close the bracket for tool_calls. The easiest way is regex
    import re
    c = re.sub(r'tool_calls=\[ToolCall\((.*?)\)\],', r'tool_calls=[ToolCall(\1)],', c)
    
    with open(p, 'w', encoding='utf-8') as f:
        f.write(c)

for f in glob.glob('tests/*.py'):
    patch_file(f)
