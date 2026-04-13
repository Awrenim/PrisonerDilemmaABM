import mesa
import os
import random
import json
from datetime import datetime
from agents import CooperatorPrisoner, DefectorPrisoner, TitForTatPrisoner, LLMPrisoner


class PrisonerModel(mesa.Model):
    """Sim enviroment for Prisoner's Dilemma"""

    def __init__(self, n_coop=3, n_defect=3, n_tft=3, n_llm=0, llm_personas=None, client=None):
        super().__init__()
        self.running = True
        self.client = client
        self.step_count = 0

        # Store run config for metadata
        self._run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._run_config = {
            'n_coop': n_coop,
            'n_defect': n_defect,
            'n_tft': n_tft,
            'n_llm': n_llm,
            'llm_personas': llm_personas or [],
            'initial_wealth': 10,
        }

        # Create logs directory and open log file
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_filename = os.path.join(log_dir, f'match_logs_{self._run_timestamp}.jsonl')
        self.log_file = open(self.log_filename, 'w', encoding='utf-8')

        # Write metadata header as first line
        metadata = {
            '_metadata': True,
            'timestamp': self._run_timestamp,
            'config': self._run_config,
        }
        self.log_file.write(json.dumps(metadata) + '\n')
        self.log_file.flush()

        # Per-step cooperation tracking
        self.current_step_coops = 0
        self.current_step_total = 0

        # Recent matches for visualization
        self.recent_matches = []
        self._max_recent_matches = 20

        self.datacollector = mesa.DataCollector(
            model_reporters={
                'Total Wealth': lambda m: sum(a.wealth for a in m.agents),
                'Active Agents': lambda m: sum(1 for a in m.agents if a.is_alive),
                'Cooperation Rate': lambda m: (
                    m.current_step_coops / m.current_step_total
                    if m.current_step_total > 0 else 0.0
                ),
            },
            agent_reporters={
                'wealth': 'wealth',
                'strategy_type': 'strategy_type'
            }
        )

        for _ in range(n_coop):
            CooperatorPrisoner(self)
        
        for _ in range(n_defect):
            DefectorPrisoner(self)
        
        for _ in range(n_tft):
            TitForTatPrisoner(self)
        
        # Handle LLM agents: via explicit personas list or n_llm count
        if llm_personas:
            for persona in llm_personas:
                LLMPrisoner(self, client=self.client, persona_type=persona)
        elif n_llm and n_llm > 0:
            available_personas = ['rationale', 'machiavellian', 'pro-social']
            for i in range(n_llm):
                persona = available_personas[i % len(available_personas)]
                LLMPrisoner(self, client=self.client, persona_type=persona)
        
    def step(self):
        self.step_count += 1

        # Reset per-step counters
        self.current_step_coops = 0
        self.current_step_total = 0
        
        active_agents = [a for a in self.agents if a.is_alive]
        
        if len(active_agents) <= 1:
            self.running = False
            return
        
        # Shuffle and pair agents for this round
        random.shuffle(active_agents)
        for i in range(0, len(active_agents) - 1, 2):
            self.resolve_match(active_agents[i], active_agents[i + 1])
        
        # Flag bankrupt agents (they stay in model.agents but won't be matched)
        for agent in active_agents:
            if agent.wealth <= 0:
                agent.is_alive = False

        # Collect data AFTER matches are resolved
        self.datacollector.collect(self)
        
    def resolve_match(self, agent_a, agent_b):

        move_a, perceived_strategy_a, rationale_a = agent_a.make_action(agent_b.unique_id)
        move_b, perceived_strategy_b, rationale_b = agent_b.make_action(agent_a.unique_id)

        if move_a == "Cooperate" and move_b == "Cooperate":
            pay_a, pay_b = 1.5, 1.5
        elif move_a == "Defect" and move_b == "Cooperate":
            pay_a, pay_b = 5, -3
        elif move_a == "Cooperate" and move_b == "Defect":
            pay_a, pay_b = -3, 5
        elif move_a == "Defect" and move_b == "Defect":
            pay_a, pay_b = -2, -2
        else:
            pay_a, pay_b = 0, 0
        
        agent_a.wealth += pay_a
        agent_b.wealth += pay_b

        agent_a.record_interaction(agent_b.unique_id, move_a, move_b)
        agent_b.record_interaction(agent_a.unique_id, move_b, move_a)

        # Track cooperation for this step
        self.current_step_total += 2
        if move_a == "Cooperate":
            self.current_step_coops += 1
        if move_b == "Cooperate":
            self.current_step_coops += 1

        # Store recent match for visualization
        match_info = {
            'step': self.step_count,
            'agent_a_id': agent_a.unique_id,
            'agent_a_strategy': agent_a.strategy_type,
            'agent_b_id': agent_b.unique_id,
            'agent_b_strategy': agent_b.strategy_type,
            'move_a': move_a,
            'move_b': move_b,
            'pay_a': pay_a,
            'pay_b': pay_b,
        }
        self.recent_matches.append(match_info)
        # Trim to keep only most recent matches
        if len(self.recent_matches) > self._max_recent_matches:
            self.recent_matches = self.recent_matches[-self._max_recent_matches:]

        log_entry = {
            'step': self.step_count,
            'agent_a_id': agent_a.unique_id,
            'agent_a_strategy': agent_a.strategy_type,
            'agent_b_id': agent_b.unique_id,
            'agent_b_strategy': agent_b.strategy_type,
            'move_a': move_a,
            'reason_a': rationale_a,
            'perceived_strategy_a': perceived_strategy_a,
            'move_b': move_b,
            'reason_b': rationale_b,
            'perceived_strategy_b': perceived_strategy_b,
            'pay_a': pay_a,
            'pay_b': pay_b,
            'wealth_a': agent_a.wealth,
            'wealth_b': agent_b.wealth,
            'is_alive_a': agent_a.is_alive,
            'is_alive_b': agent_b.is_alive
        }
        self.log_file.write(json.dumps(log_entry) + '\n')
        self.log_file.flush()

    def close_logs(self):
        """Close the log file handle when simulation is done."""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def export_data(self):
        """Export datacollector DataFrames to CSV in the logs directory."""
        log_dir = os.path.dirname(self.log_filename)

        model_df = self.datacollector.get_model_vars_dataframe()
        if not model_df.empty:
            model_path = os.path.join(log_dir, f'model_data_{self._run_timestamp}.csv')
            model_df.to_csv(model_path)
            print(f"Model data saved to {model_path}")

        agent_df = self.datacollector.get_agent_vars_dataframe()
        if not agent_df.empty:
            agent_path = os.path.join(log_dir, f'agent_data_{self._run_timestamp}.csv')
            agent_df.to_csv(agent_path)
            print(f"Agent data saved to {agent_path}")
