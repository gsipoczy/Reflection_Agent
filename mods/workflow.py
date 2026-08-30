from langgraph.graph import MessageGraph, END, StateGraph
from typing import List, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

import mods.llm as lm
import mods.prompts as prompts

def get_workflow():

    llm = lm.get_model()

    generation_prompt = prompts.generation_prompt
    generate_chain = prompts.generation_prompt | llm

    reflection_prompt = prompts.reflection_prompt
    reflect_chain = reflection_prompt | llm

    # Initialize a predefined MessageGraph
    graph = MessageGraph()

    def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
        generated_post = generate_chain.invoke({"messages": state})
        return [AIMessage(content=generated_post.content)]

    def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        # The critic must *read* the draft, not continue it. Swapping roles keeps the
        # original request as-is and turns each draft into a user turn, so the
        # conversation never ends on an assistant message (Claude 4.6+ rejects
        # assistant-message prefill).
        role_swap = {"ai": HumanMessage, "human": AIMessage}
        translated = [messages[0]] + [
            role_swap[m.type](content=m.content) for m in messages[1:]
        ]
        res = reflect_chain.invoke({"messages": translated})
        return [HumanMessage(content=res.content)]

    def should_continue(state: List[BaseMessage]):
        print(state)
        print(len(state))
        print("----------------------------------------------------------------------")
        if len(state) > 6:
            return END
        return "reflect"

    graph.add_node("generate", generation_node)
    graph.add_node("reflect", reflection_node)

    graph.add_edge("reflect", "generate")

    graph.set_entry_point("generate")

    graph.add_conditional_edges("generate", should_continue)

    workflow = graph.compile()

    return workflow

