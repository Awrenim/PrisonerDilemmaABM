# Prisoner’s Dilemma: Agent-Based Modelling with LLM Personas

## Introduction

This project is dedicated to an agent-based model (*ABM*) exploring how different LLM-driven agents with distinct personas behave in the iterated Prisoner’s Dilemma Game. 

The iterated Prisoner’s Dilemma is one of the most studied problems in game theory and behavioral science. It has been studied as a purely theoretical construct and utilized in many daily social interactions, even TV shows (i.e. the British *Golden Balls* show). There are well-known and described strategies in literature \- *Always Cooperate*, *Always Defect* or *Tit-for-Tat*. But what happens when we introduce LLM agents that **evaluate** their opponents?

This project aims to investigate whether LLM agents, given different persona prompts, develop meaningfully different behavioral patterns and how those patterns interact with classical strategies in a competitive environment with real consequences. 

## The Game \- model and settings

The game world is based on Prisoner’s Dilemma (obviously). Every agent starts with **10 gold** in its pocket. Each round, agents are paired in a match where they individually decide to **Cooperate** or **Defect**. Depending on both agents’ decisions the outcome of the match is as below:

|  | Agent A: cooperates | Agent B: defects |
| :---- | :---- | :---- |
| Agent B: cooperates | \+ 1.5 / \+ 1.5 | \- 3 / \+ 5 |
| Agent B: defects | \+ 5 / \- 3 | \- 2 / \- 2 |

If the agent's wealth drops to 0 or below, the agent goes bankrupt and is permanently removed from the game.

## Agents

In the project there are two types of agent: hard-coded strategies and LLMs with personas.

**Classical (hard-coded) strategies**:

- *Cooperator:* always cooperates.  
- *Defector*: always defects.  
- *Tit-for-tat*: cooperates first, then mirrors opponent’s last action

**LLM Personas**:

- *Rational*: analytical, game-theory oriented. Aims for optimal expected value based on pattern recognition.  
- *Machiavellian*: self-interested and exploitative. Views cooperation as a tool for manipulation.  
- *Pro-social*: values mutual benefit and trust-building. Reluctant to defect even when advantageous.

LLM is fed with a system prompt consisting of game rules, structured output JSON schema rule and a task prompt describing agent’s persona, opponent’s ID (but **not** their strategy), current wealth and match history with this particular opponent. 

## Project structure and technical aspects

This project utilizes [Mesa](https://github.com/projectmesa/mesa) agent-based modeling framework and its scheduler to orchestrate agent matches. For LLM agents I use OpenAI SDK to make API calls \- the selected model is *gpt-5.4-nano*.

Default settings for model: 

- 12 agents (3x *Cooperator*, 3x *Defector*, 3x *Tit-for-Tat*, 1x *Rational*, 1x *Machiavellian*, 1x *Pro-social*)  
- Payoff matrix as mentioned above  
- Starting wealth \= 10  
- Max rounds \= 200

Project directory structure:  
\- agents.py  			\# Agent classes (base class, classical and LLM strategies)  
\- model.py  			\# Simulation environment, matching scheduler, payoffs  
\- PrisonersDilemma.py	\# Main file, CLI entry point for batch runs  
\- requirements.txt		\# Self-explanatory  
\- .env				\# “OPENAI\_API\_KEY” value

To run simulation, simply paste the API key value in .env file and exe the main file (*PrisonersDilemma.py*).   
You can find the configuration variables (*n\_{strategy\_agent}*, max rounds as *n\_steps* etc.) to suit personal preferences.

The project comes with example run logs (default simulation settings, gpt-5.4-nano) to look up (./log directory).

## Logging and data collector

A big value of this simulation is data collection. It’s split into two channels:

1. \[Macro\] Mesa internal Data Collector: tracking model- and agent-scope data.  
   1. model\_data\_\*:*Total Wealth, Active Agents, Cooperation Rate;*  
   2. agent\_data\_\*: per-agent wealth in every round.  
2. \[Micro\] JSONL Match logs \- granular per-match records. Written directly to a .jsonl file during matches. Each entry is a full match information:  
   1. step, agent IDs, strategy types;  
   2. each agent’s move, reasoning and perceived opponent strategy;  
   3. payoffs;  
   4. post-match wealth and alive status.

Both channels are written to ./log directory.

## Additional comments

- The current payout matrix is quite aggressive and encourages behavior that betrays the opponent.  
- LLM personas are simple, extensible prompts. In their current state, they exhibited behavior similar to that of their corresponding hardcoded strategies. For example, when tested with a more lenient payoff matrix, longer games, and even weaker models (e.g., GPT-4o Mini), the final results and dynamics of the Pro-Social persona were similar to those of the hardcoded Cooperator strategy.  
- With that in mind, it would be a good idea to use psychometric frameworks when creating personas, such as [\[2307.00184\] Personality Traits in Large Language Models](https://arxiv.org/abs/2307.00184)  
- In future efforts to utilize or expand this simulation, data collectors should be fully leveraged—in particular, to analyze how LLMs reason when making decisions.