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
from util import Queue
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        foods = newFood.asList()
        foodDistances = []                                                
        evaluation =  0
        capsules = successorGameState.getCapsules()
        capsuleDistances = []

        # print(str(newFood))
        # print(str(foods))

        
        if(newPosition == currentGameState.getPacmanPosition()):
            return (-float("inf"))
        
        if successorGameState.isLose():
            return -float("inf")
        
        if successorGameState.isWin():
            return float ('inf')
        
        for food in foods:
            foodDistances.append(mazeDistance(food,newPosition,currentGameState))
        
        for capsule in capsules:
            capsuleDistances.append(manhattanDistance(capsule,newPosition))
            
        for ghost in newGhostStates:
            if(manhattanDistance(newPosition,ghost.getPosition()) <=50):
                if (ghost.scaredTimer == 0 and manhattanDistance(newPosition,ghost.getPosition()) <=5):
                    evaluation += 500 / (min(capsuleDistances, default=0) + 1)
                    if manhattanDistance(newPosition,ghost.getPosition()) <=1:
                        return -float("inf")
                # if (ghost.scaredTimer == 0 and manhattanDistance(newPosition,ghost.getPosition()) <=1):
                #     return -float('inf')
                elif ghost.scaredTimer != 0:
                    evaluation += 2000 / (manhattanDistance(newPosition,ghost.getPosition()) +1)
                    if(manhattanDistance(newPosition,ghost.getPosition()) == 0):
                        return float('inf')
            
        if(currentGameState.getNumFood() > successorGameState.getNumFood()):
            evaluation += 1200 
        evaluation += 500/(min(foodDistances) + 1) + 600 / (sum(foodDistances) + 1)
        return evaluation
    
    
def mazeDistance(start, end, gameState):
    """
    Returns the distance between start and end in the Pacman game maze.
    """
    walls = gameState.getWalls()
    visited = set()
    queue = Queue() 
    queue.push((start, 0))  

    while not queue.isEmpty():
        position, distance = queue.pop() 
        x, y = position
        
        if position == end:
            return distance 
        

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            new_position = (new_x, new_y)
            
            if not walls[new_x][new_y] and new_position not in visited:
                visited.add(new_position)  
                queue.push((new_position, distance + 1))  
                
    return float('inf') 



def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        # util.raiseNotDefined()
        _, action = self.value(gameState, 0, self.depth)
        return action

    def value(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        elif agentIndex > 0:
            return self.minValue(gameState, agentIndex, depth)
        
    def minValue(self, gameState, agentIndex, depth):
        currValue, currAction = 1e9, None

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore, _ = self.value(successorGameState, nextAgent, nextDepth)

            # currValue = min(currValue, successorScore)
            if successorScore < currValue:
                currValue = successorScore
                currAction = action

        return currValue, currAction
    
    def maxValue(self, gameState, agentIndex, depth):
        currValue, currAction = -1e9, None

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)

            successorScore, _ = self.value(successorGameState, nextAgent, nextDepth)

            # currValue = min(currValue, successorScore)
            if successorScore > currValue:
                currValue = successorScore
                currAction = action

        return currValue, currAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        alpha=-(float("inf"))
        beta=float("inf")

        _, action = self.value(gameState, 0, self.depth,alpha,beta)
        return action

    def value(self, gameState, agentIndex, depth, alpha,beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth,alpha,beta)
        elif agentIndex > 0:
            return self.minValue(gameState, agentIndex, depth,alpha,beta)
        
    def minValue(self, gameState, agentIndex, depth,alpha,beta):
        currValue, currAction = 1e9, None

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore, _ = self.value(successorGameState, nextAgent, nextDepth,alpha,beta)

            # currValue = min(currValue, successorScore)
            if successorScore < currValue:
                currValue = successorScore
                currAction = action
                if currValue < alpha: 
                    return currValue, currAction
                beta = min(beta, currValue)


        return currValue, currAction
    
    def maxValue(self, gameState, agentIndex, depth,alpha,beta):
        currValue, currAction = -1e9, None

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)

            successorScore, _ = self.value(successorGameState, nextAgent, nextDepth,alpha,beta)

            # currValue = min(currValue, successorScore)
            if successorScore > currValue:
                currValue = successorScore
                currAction = action
                if currValue > beta:
                    return currValue, currAction
                alpha = max(alpha,currValue)
                
        return currValue, currAction
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.value(gameState, 0, self.depth)
        return action

    def value(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        elif agentIndex > 0:
            return self.expValue(gameState, agentIndex, depth)
        
    def expValue(self, gameState, agentIndex, depth):
        currValue, currAction = 0, None
        prob =  1/len(gameState.getLegalActions(agentIndex))
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore, _ = self.value(successorGameState, nextAgent, nextDepth)

            # currValue = min(currValue, successorScore)
            # if successorScore < currValue:
            currValue += prob * successorScore
            currAction = action

        return currValue, currAction
        
    def maxValue(self, gameState, agentIndex, depth):
        currValue, currAction = -1e9, None

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)

            successorScore, _ = self.value(successorGameState, nextAgent, nextDepth)

            # currValue = min(currValue, successorScore)
            if successorScore > currValue:
                currValue = successorScore
                currAction = action

        return currValue, currAction
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    currentPosition = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    foodDistances = []   
    evaluation = score                                             
    for food in foodPositions:
        foodDistances.append(mazeDistance(food,currentPosition,currentGameState))
    
    for ghost in ghostStates:
        if (manhattanDistance(currentPosition, ghost.getPosition()) <= 2 and ghost.scaredTimer == 0):
            return -float('inf')
        elif (ghost.scaredTimer > 0 and manhattanDistance(currentPosition, ghost.getPosition()) <= 10):
            evaluation += 20000 / (manhattanDistance(currentPosition, ghost.getPosition()) +1)
        if ghost.scaredTimer > 0:
            evaluation += 500 
            
    if (currentGameState.isWin()):
        return float('inf')
    
    currentActions = currentGameState.getLegalActions(0)
    
    # for action in currentActions:
    #     successorGameState = currentGameState.generatePacmanSuccessor(action)
    #     newPosition = successorGameState.getPacmanPosition()
    #     if newPosition in foodPositions:
    #         evaluation += 10
    
    # if currentPosition not in foodPositions:
    #     evaluation -=1000 
    #     print("not in food")
    #     print(currentPosition)
    #     print(foodPositions)
    
    # evaluation +=  800/(currentGameState.getNumFood()+1) + 200/(min(foodDistances, default=0)+1)
    evaluation -= -150/(min(foodDistances, default=0) + 1) + 50*(currentGameState.getNumFood())
    
    
    
    return evaluation

# Abbreviation
better = betterEvaluationFunction
