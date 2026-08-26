from app.evaluation.benchmark_dataset import BenchmarkDataset
from app.evaluation.benchmark_runner import BenchmarkRunner
from app.agent.agent import Agent
from app.agent.registry import AgentRegistry
from app.agent.routing import CapabilityRouter
from app.llm.gateway import ModelGateway, ModelRouter, ModelProfile, ModelCapability
from app.llm.provider import ModelProvider
from app.llm.models import ModelRequest, ModelResponse

class DummyProvider(ModelProvider):
    @property
    def provider_id(self): return "dummy"
    def generate(self, req, model):
        return ModelResponse(content="Paris", input_tokens=10, output_tokens=5, total_tokens=15, model=model, provider="dummy")

ds = BenchmarkDataset.load_jsonl("datasets/eval_starter.jsonl")

# Minimal agent setup
registry = AgentRegistry()
router = CapabilityRouter()
profiles = [ModelProfile(provider="dummy", model_id="dummy-1", capabilities={ModelCapability.TEXT_GENERATION})]
gateway = ModelGateway(router=ModelRouter(profiles), providers={"dummy": DummyProvider()})
agent = Agent(provider=gateway, registry=registry)

runner = BenchmarkRunner(agent)
res = runner.run(ds, {"model_id": "dummy-1"})
print(f"Success rate: {res.summary['success_rate']}")
print(f"Avg tokens: {res.case_results[0]['operational']['total_tokens']}")
