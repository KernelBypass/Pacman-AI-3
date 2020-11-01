# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

debug_Print = False
it = 0


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        stateSpace = self.mdp.getStates()

        # Value iterations
        for iteration in range(self.iterations):
            if debug_Print:
                "\n\n\n+++ Value Iteration #", iteration, " +++\n"
            # Create temp value dictionary [for the "batch" version of value iteration--to preserve self.values
            # unaltered between iterations for the computeQValueFromValues() subroutine calls]
            iterValueSpace = util.Counter()
            # Traverse states ("batch" version iteration)
            for state in stateSpace:
                # Compute best actions for states where actions exist (states that are not terminal)
                if not self.mdp.isTerminal(state):
                    # Extract possible actions for the current state
                    actions = self.mdp.getPossibleActions(state)
                    # Iterate through actions and capture the q-value of the highest-scoring action
                    bestActionValue = max(self.getQValue(state, action) for action in actions)
                    # Add this value to the state-value space
                    iterValueSpace[state] = bestActionValue
            # Save the results of a complete value iteration after having finished traversing the states
            # I.e., "full backup" (per Sutton & Barto)
            self.values = iterValueSpace



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # Initialize q-Value accumulator variable
        qVal = 0.0

        #### Debug printing module #######
        global it
        if debug_Print:
            print "\n\n\nCommencing qVal call #", it, "\n******************\n"
        it += 1
        f = 0
        ##############END DEBUG PRINT###############


        # Traverse available states and probabilities of getting to them via transition function call
        for (newState, probability) in self.mdp.getTransitionStatesAndProbs(state, action):

            #### Debug printing module #######
            if debug_Print:
                print "\nFor loop iteration #", f, "\nnewState: ", newState, "\nProbability: ", probability, "\nQVal: ", qVal
            #########END DEBUG PRINT###########
            value = self.values[newState]
            reward = self.mdp.getReward(state, action, newState)
            gamma = self.discount
            qVal = qVal + probability * (reward + gamma * value)

            # Debug counter #
            f+=1
            #################

        return qVal



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # If no legal actions, return None
        if self.mdp.isTerminal(state):
            return None
        # Accumulator of action-value pairs
        actions = dict()
        # All possible actions for a given state
        actionSpace = self.mdp.getPossibleActions(state)

        # For the tie-breaking magic (not really needed for this question, but making the code reusable for later question)
        import random as random

        # Traverse all possible actions
        for action in actionSpace:
            # And put every pair of action and corresponding Q-Value in the accumulator dictionary
            actions[action] = self.getQValue(state, action)

        # Get max value
        bestVal = max(actions.values())
        # Break a potential tie randomly
        return random.choice([k for (k, v) in actions.items() if v == bestVal])



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
