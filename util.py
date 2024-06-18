import textwrap

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""When you respond to a question, please use the language from the reference passage provided below.
                           Responses should be in complete sentences, summarizing all relevant information. 
                           As the audience has technical expertise, keep the answers concise and clear; 
                           if there are multiple points, address each one distinctly. 
                           If the passage is irrelevant to the question, you may omit it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt