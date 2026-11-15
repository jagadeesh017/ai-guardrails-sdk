from .agent import register_agent_rules
from .input import register_input_rules
from .ops import register_ops_rules
from .output import register_output_rules
from .rag import register_rag_rules


def register_all_rules(registry):
    register_input_rules(registry)
    register_rag_rules(registry)
    register_agent_rules(registry)
    register_output_rules(registry)
    register_ops_rules(registry)