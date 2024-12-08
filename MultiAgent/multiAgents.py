# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        if Directions.STOP in legalMoves and len(legalMoves) > 1:
            legalMoves.remove(Directions.STOP)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        "*** YOUR CODE HERE ***"
        food = newFood.asList()
        foodDistances = [manhattanDistance(newPos,eatstuff) for eatstuff in food]
        ghostPositions = successorGameState.getGhostPositions()
        priorityAdjustment = 0 

        for i in foodDistances:
            if i <= 4:
                priorityAdjustment += 1
            elif i > 4 and i <= 15:
                priorityAdjustment += 0.5
            else:
                priorityAdjustment += 0.25

        for ghost in ghostPositions:
            if ghost == newPos: 
                priorityAdjustment = 2 - priorityAdjustment

            elif manhattanDistance(ghost,newPos) <= 3.5:
                priorityAdjustment = 1 - priorityAdjustment

        return successorGameState.getScore() + priorityAdjustment

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_agent(state: GameState, depth: int):
            if (state.isWin() or state.isLose()):
                return state.getScore()
            
            pac_actions = state.getLegalActions(0)
            best_score = -10000000000000000000;
            for action in pac_actions:
                score = exp_agent(state.generateSuccessor(0, action), depth, 1) #exp on the next pacman action
                if (score > best_score):
                    best_score = score
                    best_pac_action = action
            if (depth == 0):
                return best_pac_action #This would only run if
            else: 
                return best_score   
            
        
        def exp_agent(state: GameState, depth: int, ghost_index: int):
            if (state.isWin() or state.isLose()):
                return state.getScore()
            
            last_ghost_index = state.getNumAgents() - 1 #get number of ghost in the game
            best_score = 10000000000000000000
            actions = state.getLegalActions(ghost_index)
            for action in actions:
                if (ghost_index == last_ghost_index):
                    if (depth == self.depth - 1): #this is the leaf of the minimax tree
                        score = self.evaluationFunction(state.generateSuccessor(ghost_index, action))
                    else: #Get to the next depth
                        score = max_agent(state.generateSuccessor(ghost_index, action), depth + 1)
                else:
                    score = exp_agent(state.generateSuccessor(ghost_index, action), depth, ghost_index + 1) #Continue to the next ghost action
                    
                if (score < best_score):
                    best_score = score  #best_score needs to be the smallest ones because we have to find the "best" action not the "highest score possible" action.
                                        #This algorithm assumes the ghosts would also makes the best action. So the best score would be the smallest score in that sub gametree
            return best_score
        return max_agent(gameState, 0) #Current base case

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # Điều kiện dừng: trạng thái kết thúc hoặc đạt độ sâu tối đa
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman (người chơi tối đa)
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:  # Ghosts (người chơi tối thiểu)
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            """
            Hàm tính giá trị tối đa cho Pacman.
            """
            value = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, alphaBeta(1, depth, successor, alpha, beta))
                if value > beta:  # Cắt tỉa
                    return value
                alpha = max(alpha, value)
            return value

        def minValue(agentIndex, depth, gameState, alpha, beta):
            """
            Hàm tính giá trị tối thiểu cho Ghosts.
            """
            value = float("inf")
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = min(value, alphaBeta(nextAgent, nextDepth, successor, alpha, beta))
                if value < alpha:  # Cắt tỉa
                    return value
                beta = min(beta, value)
            return value

        # Tìm hành động tốt nhất cho Pacman (agentIndex = 0)
        alpha, beta = float("-inf"), float("inf")
        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(0):  # Pacman luôn là agent 0
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(1, 0, successor, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex): 
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if agentIndex == 0: #Pacman's turn (maximizing player)
                return maxValue(state,depth)
            else:
                return expValue(state,depth, agentIndex)
            
        def maxValue(state, depth):
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v,expectimax(state.generateSuccessor(0, action), depth,1))
            return v
        
        def expValue(state, depth, agentIndex):
            v = 0 
            actions = state.getLegalActions(agentIndex)
            prob = 1/len(actions)
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                depth += 1
            for action in actions:
                v += prob * expectimax(state.generateSuccessor(agentIndex,action), depth, nextAgent)
            return v
        

        #Start Expectimax Search
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = expectimax(gameState.generateSuccessor(0, action),0,1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    remainingCapsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()
    capsuleScoreAdjustment = 0

    # Current Pacman Position
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    # Capsule
    if remainingCapsules:
        for capsule in remainingCapsules:
            distanceToCapsule = manhattanDistance(capsule, pacmanPosition)
            if distanceToCapsule > 0:
                capsuleScoreAdjustment += 1.0 / distanceToCapsule
            else:
                capsuleScoreAdjustment -= 100  

    # Ghost
    ghostDistanceAdjustment = 0
    for ghost in ghostStates:
        ghostPosition = ghost.getPosition()
        distanceToGhost = manhattanDistance(pacmanPosition, ghostPosition)
        ghostDistanceAdjustment += 1.0 / (1 + distanceToGhost)  

    return currentScore + capsuleScoreAdjustment - ghostDistanceAdjustment

# Abbreviation
better = betterEvaluationFunction
