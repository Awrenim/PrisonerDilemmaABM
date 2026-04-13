import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from model import PrisonerModel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def main():
    """Run the Prisoner's Dilemma simulation."""
    # Configuration
    N_COOP = 3
    N_DEFECT = 3
    N_TFT = 3
    LLM_PERSONAS = ['rationale', 'machiavellian', 'pro-social']  # One LLM agent per persona
    N_STEPS = 200

    model = PrisonerModel(
        n_coop=N_COOP,
        n_defect=N_DEFECT,
        n_tft=N_TFT,
        llm_personas=LLM_PERSONAS,
        client=client
    )

    print(f"Starting simulation: {N_COOP} Cooperators, {N_DEFECT} Defectors, {N_TFT} TFT, {len(LLM_PERSONAS)} LLM {LLM_PERSONAS}")
    print(f"Running for {N_STEPS} steps...\n")

    start_time = time.time()

    for step in range(N_STEPS):
        if not model.running:
            print(f"\nSimulation ended early at step {step} (too few agents remaining).")
            break
        
        active_before = {a.unique_id for a in model.agents if a.is_alive}
        n_matches = len(active_before) // 2
        print(f"Step {step + 1}/{N_STEPS} | Active agents: {len(active_before)} | Matches: {n_matches}")
        
        model.step()
        
        # Report bankruptcies this round
        for a in model.agents:
            if a.unique_id in active_before and not a.is_alive:
                print(f"  >> Agent {a.unique_id} ({a.strategy_type}) went bankrupt! Final wealth: {a.wealth:.1f}")

    # Summary
    print("\n=== Simulation Results ===")
    print(f"Steps completed: {model.step_count}")

    all_agents = list(model.agents)
    active = [a for a in all_agents if a.is_alive]

    print(f"Agents still active: {len(active)}\n")
    print(f"{'ID':<5} {'Strategy':<20} {'Wealth':>10} {'Status':<10}")
    print("-" * 50)
    for agent in sorted(all_agents, key=lambda a: a.unique_id):
        status = "Active" if agent.is_alive else "Bankrupt"
        print(f"{agent.unique_id:<5} {agent.strategy_type:<20} {agent.wealth:>10.1f} {status:<10}")

    # Export datacollector data and close log file
    model.export_data()
    model.close_logs()
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nMatch logs saved to {model.log_filename}")
    print(f"Simulation ran for {minutes} minutes {seconds} seconds.")


if __name__ == "__main__":
    main()
