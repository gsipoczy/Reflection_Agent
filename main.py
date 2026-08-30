from langchain_core.messages import HumanMessage

import mods.workflow as wf

workflow = wf.get_workflow()

inputs = HumanMessage(content="""Write a linkedin post on getting a software developer job at IBM under 160 characters""")

response = workflow.invoke(inputs)

print(response[-1].content)


