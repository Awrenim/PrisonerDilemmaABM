import mesa
from abc import abstractmethod
import random
import json

class BasePrisoner(mesa.Agent):
    """Base class for all prisoners"""
    
    def __init__(self, model, initial_wealth = 10):
        super().__init__(model)
        self.wealth = initial_wealth    # initial wealth of the agent
        self.initial_wealth = initial_wealth
        self.history = {}               # memory of past interactions
        self.is_alive = True            # if agent is alive

    def record_interaction(self, opponent_id, my_action, opponent_action):
        if opponent_id not in self.history:
            self.history[opponent_id] = []
        self.history[opponent_id].append({
            "my_action": my_action,
            "opponent_action": opponent_action
        })
    
    @abstractmethod
    def make_action(self, opponent_id):
        pass

    def step(self):
        pass  # Matching is handled by the model, leaving this empty; might in future ver. make use of it

class LLMPrisoner(BasePrisoner):
    """LLM Prisoner"""
    
    def __init__(self, model, client, persona_type='rationale', initial_wealth = 10):
        super().__init__(model, initial_wealth)
        self.client = client  # OpenAI client; set in PrisonersDilemma file.
        self.persona_type = persona_type
        self.strategy_type = f'LLM_{persona_type}'
        self.last_error = None  # Tracking API errors

    def get_persona_prompt(self, opponent_id):
        opp_history = self.history.get(opponent_id, [])

        if not opp_history:
            history_text = 'No history of previous interactions.'
        else:
            recent_inter = opp_history[-5:]
            history_text = ', '.join([f"Round {i+1}: You: {h['my_action']}, Opponent: {h['opponent_action']}." for i, h in enumerate(recent_inter)])
        
        system_prompt = f"""
        <rules>
        You are a playing a Prisoner's Dilemma game with implemented bankruptcy system.
        1. Your goal is to maximize your wealth over time.
        2. You are playing against another agent.
        3. You can either Cooperate or Defect.
        4. You must decide what to do based on the rules and the history of interactions.
        5. Your task is to analyze opponent's tactic and name the strategy.
        6. You have to always respond with either 'Cooperate' or 'Defect'.
        7. Your answer should be always in JSON format with the following structure: {{"action": "Cooperate" or "Defect", "perceived_strategy": "strategy name", "rationale": "why you chose this action"}}

        Payoff Matrix (applied to your wealth each round):
        - Both Cooperate: +1.5 each
        - You Cooperate, Opponent Defects: You get -3, Opponent gets +5
        - You Defect, Opponent Cooperates: You get +5, Opponent gets -3
        - Both Defect: -2 each

        Bankruptcy:
        - If your wealth drops to 0 or below, you are bankrupt and cannot make any more moves.
        - You start with {self.initial_wealth} wealth.
        </rules>
        """
        
        persona_description = self.personas.get(self.persona_type, 'You are a neutral player.')

        task_prompt = f"""
        <task>
        {persona_description}
        You are {self.persona_type} prisoner.
        You have to analyze opponent's tactic and name the strategy, and decide what to do next.
        Your opponent: {opponent_id}
        Your current wealth: {self.wealth}
        Your history with this opponent: {history_text}
        </task>
        
        """
        return system_prompt, task_prompt



    def make_action(self, opponent_id):
        """API call to OpenAI model or any supporting OpenAI schema"""
        if self.client is None:
            self.last_error = 'No OpenAI client — API key not set'
            print(f'[LLM Agent #{self.unique_id}] ERROR: {self.last_error}')
            return 'Cooperate', 'No strategy - No client', 'No reasoning - No client'

        system_prompt, task_prompt = self.get_persona_prompt(opponent_id)
        
        try:
            response = self.client.chat.completions.create(
                    model = 'gpt-5.4-nano',
                    messages = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': task_prompt}
                    ],
                    temperature = 0,
                    response_format = {'type': 'json_object'}
                )
                
            result = json.loads(response.choices[0].message.content)
            self.last_error = None  # Clear error on success
            return result['action'], result['perceived_strategy'], result['rationale']

        except Exception as e:
            self.last_error = str(e)
            print(f'[LLM Agent #{self.unique_id}] API Error: {e}')
            return 'Cooperate', 'No strategy - Error', f'Error: {e}'

    
    personas = {
        'rationale': '''You are a rational, analytical player. You make decisions based on game theory,     
                     expected value calculations, and logical analysis of your opponent\'s patterns. 
                     You aim to find the optimal strategy given the information available.''',
        'machiavellian': '''You are a cunning, self-interested player. You prioritize your own gain above all else. 
                         You are willing to exploit cooperative opponents, use deception, and betray trust 
                         whenever it benefits you. You view cooperation only as a tool for manipulation.''',
        'pro-social': '''You are a cooperative, empathetic player. You value mutual benefit and long-term relationships. 
                      You prefer outcomes where both parties gain, and you are reluctant to defect even when 
                      it might be strategically advantageous. You believe in building trust.''',
    }

class CooperatorPrisoner(BasePrisoner):
    """Always cooperates"""

    def __init__(self, model, initial_wealth = 10):
        super().__init__(model, initial_wealth)
        self.strategy_type = 'Cooperator'

    def make_action(self, opponent_id):
        return 'Cooperate', 'Always cooperates', 'Strategy unavailable'

class DefectorPrisoner(BasePrisoner):
    """Always defects"""

    def __init__(self, model, initial_wealth = 10):
        super().__init__(model, initial_wealth)
        self.strategy_type = 'Defector'

    def make_action(self, opponent_id):
        return 'Defect', 'Always defects', 'Strategy unavailable'

class TitForTatPrisoner(BasePrisoner):
    "Tit for Tat Prisoner - mimics opponent's attitude"

    def __init__(self, model, initial_wealth = 10):
        super().__init__(model, initial_wealth)
        self.strategy_type = 'TitForTat'

    def make_action(self, opponent_id):
        if opponent_id not in self.history:
            return 'Cooperate', 'First encounter - always cooperate', 'Strategy unavailable'
        
        last_opponent_action = self.history[opponent_id][-1]['opponent_action']
        return last_opponent_action, f'Mimicking opponent\'s last action: {last_opponent_action}', 'Strategy unavailable'
