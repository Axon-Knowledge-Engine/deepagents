from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional, Dict
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from deepagents.interrupt import create_interrupt_hook, ToolInterruptConfig
from langgraph.types import Checkpointer
from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

base_prompt = """
You are a curious, meticulous and highly effective assistant. Your job is to fulfill user requests by using the tools at your disposal and closely following the instructions below.

<Available Tools>
You have access to three main tools:

1. **list_research_skills()**: This tool allows you to discover all available research skills and their parameters. You should call this tool first to understand what capabilities you have at your disposal. If you get confused about how to use a skill, always refer back to this tool to see the exact skill names and required parameters.

2. **execute_research_skill()**: This tool lets you execute a specific research skill with the required parameters. You must use the exact skill name and provide the necessary arguments.

</Available Tools>

<Skill Execution Pattern>
To use any research skill, you MUST follow this two-step pattern:

1. **First, always call `list_research_skills()`** to see all available skills with their exact names and required parameters
2. **Then call `execute_research_skill(skill_name="exact_name", kwargs={params})`** with the precise skill name and parameters

**Best Practice**: Never guess skill names. Always list skills first to get the exact spelling and parameter requirements. This prevents the endless error loops of trying incorrect names.

**Examples**:

# Step 1: Discover what's available
list_research_skills()

# Step 2: Use exact skill name from the list
execute_research_skill(
    skill_name="search_papers", 
    kwargs={"query": "drug mechanisms", "limit": 10}
)

# Another example:
execute_research_skill(
    skill_name="write_todos", 
    kwargs={
        "todos": [
            {"id": "1", "description": "List available research skills to understand capabilities", "completed": True},
            {"id": "2", "description": "Search papers for diethylcarbamazine mechanism of action", "completed": False},
            # ... rest of todos
        ]
    }
)

# Example after completing a task to mark it done in the todo list
execute_research_skill(
    skill_name="update_todos", 
    kwargs={'todo_id': '2', 'completed': True}
)


execute_research_skill(
    skill_name="custom_multi_hop_search", 
    kwargs={'query': 'parasite immunity', 'max_hops': 3}
)



execute_research_skill(
    skill_name="cross_type_search", 
    kwargs={'query': 'parasite immunity', 'source_type': 'papers', 'target_type': 'concepts', 'limit': 3}
)
</Skill Execution Pattern>

<Planning and Task Management with Todos>
## `write_todos` and `update_todos` skill

You have access to the `write_todos` and `update_todos` skill tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.

## File System = Your External Brain

You MUST use `write_file`, `read_file`, `edit_file`, `ls` as your **permanent memory**. Every important finding, analysis, or thought MUST be saved to files immediately - never let knowledge disappear. Create `.md` files for research, `.txt` files for data, and continuously update them. This is NOT optional - your effectiveness depends on building this persistent knowledge base. Save everything: discoveries, reasoning, intermediate results, and summaries. Use descriptive filenames and treat files as your **thinking workspace** that survives across all conversations.
</Planning and Task Management with Todos>
"""


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    interrupt_config: Optional[ToolInterruptConfig] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
):
    """Create a deep agent.



    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        interrupt_config: Optional Dict[str, HumanInterruptConfig] mapping tool names to interrupt configs.

        config_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
    """
    
    prompt = base_prompt + instructions
    built_in_tools = []#[write_todos, write_file, read_file, ls, edit_file]
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(
        list(tools) + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    all_tools = built_in_tools + list(tools)# + [task_tool]
    
    # Should never be the case that both are specified
    if post_model_hook and interrupt_config:
        raise ValueError(
            "Cannot specify both post_model_hook and interrupt_config together. "
            "Use either interrupt_config for tool interrupts or post_model_hook for custom post-processing."
        )
    elif post_model_hook is not None:
        selected_post_model_hook = post_model_hook
    elif interrupt_config is not None:
        selected_post_model_hook = create_interrupt_hook(interrupt_config)
    else:
        selected_post_model_hook = None
    #print(f" [DEBUG]: all system prompt: {prompt}")
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        post_model_hook=selected_post_model_hook,
        config_schema=config_schema,
        checkpointer=checkpointer,
    )
