from langchain.agents import AgentExecutor, Tool
from langchain.chains import LLMChain
from langchain.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
import retrieval as re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue, MatchAny
import api
import numpy as np

def retrieve_country_policies(country_name: str, query: str, collection, year_threshold:int, k=50):
    years = [str(year) for year in range(year_threshold, 2100)]

    filter = Filter(
        must=[
            FieldCondition(
                key="country",
                match=MatchValue(value=country_name)
            ),
            FieldCondition(
                key="year",
                match=MatchAny(any=years),
            )
        ]
    )

    retriever = re.Retriever(query, collection)
    retriever.similarity_search_with_filter(k, filter)
    # retriever.rerank(['policy', 'effect'])
    # retriever.cos_filtering(['policy', 'effect'], 0.8, 30)
    retriever.filtered_countents = retriever.found_docs
    context = " "
    for filtered_context in retriever.filtered_contents:
        context += filtered_context
    
    return context

def retrieve_knowledge(query: str, collection, k=50):
    retriever = re.Retriever(query, collection)
    retriever.similarity_search(k)
    # retriever.rerank('content')
    # retriever.cos_filtering('content', 0.8, 30)

    context = " "
    for filtered_context in retriever.filtered_contents:
        context += filtered_context
    
    return context

class CountryAgent:
    def __init__(self, country_name, llm, stance=None):
        self.country_name = country_name
        self.llm = llm
        self.policy_memory = []
        self.stance = stance if stance else "balanced approach"

    def propose_policy(self, shared_goal, policy_collection, knowledge_collection, year):
        retrieved_policies = retrieve_country_policies(self.country_name, shared_goal, policy_collection, year)
        retrieved_knowledges = retrieve_knowledge(shared_goal, knowledge_collection)

        prompt = f"""
        You are the policy advisor for {self.country_name}, which has a {self.stance}.
        
        Shared goal: {shared_goal}
        
        Historical policies for {self.country_name}:
        {retrieved_policies}

        Relevant knowledge base items:
        {retrieved_knowledges}

        Based on your country's interests and history, propose a new climate policy 
        for {self.country_name} that aligns with the shared goal. 
        You may also highlight any points of potential contention or 
        unique considerations for {self.country_name}.
        """
        response = self.llm(prompt)
        self.policy_memory.append(response.content)
        return response.content
    
    def react_to_other_policies(self, other_policies):
        prompt = f"""
        You are {self.country_name}'s policy advisor, with a {self.stance}.
        
        Other countries have proposed the following policies:
        {other_policies}

        Your task:
        1. Critique or debate these proposals from the perspective of {self.country_name}.
           Identify any conflicts, disagreements, or potential synergies.
        2. If needed, refine or adjust your own policy to protect or promote your 
           country's interests and approach.
        3. Provide a clear statement of how your revised policy stands in contrast 
           or alignment with the others.
        """
        response = self.llm(prompt)
        self.policy_memory.append(response)
        return response

def multi_agent_climate_discussion(countries, shared_goal, policy_collection, knowledge_collection, year, stances=None):
    """
    stances: A dictionary mapping each country to a specific stance string (optional).
             Example: {'USA': 'strong oil & gas interests', 'France': 'nuclear energy focus'}
    """
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7, api_key=api.OPENAI_API)

    agents = []
    for country in countries:
        stance = stances.get(country, None) if stances else None
        agents.append(CountryAgent(country, llm, stance=stance))

    proposals = {}
    for agent in agents:
        proposals[agent.country_name] = agent.propose_policy(shared_goal, policy_collection, knowledge_collection, year)

    for agent in agents:
        other_proposals_text = "\n".join(
            f"{cntry} proposed: {prop}" for cntry, prop in proposals.items() if cntry != agent.country_name
        )
        reaction = agent.react_to_other_policies(other_proposals_text)
        proposals[agent.country_name] += "\n\nREVISED PROPOSAL:\n" + reaction.content

    result_dict = {}
    for country, final_policy_text in proposals.items():
        result_dict[country] = final_policy_text

    return result_dict