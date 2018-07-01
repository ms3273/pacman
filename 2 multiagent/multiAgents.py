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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores      = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore   = max(scores)
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
        newPos             = successorGameState.getPacmanPosition()
        newFood            = successorGameState.getFood()
        
        # Find the food positions
        min_dist = 10000
        for i,food_col in enumerate(newFood):
          for j,val in enumerate(food_col):
            if val: 
              x,y  = newPos
              dist = abs(x-i) + abs(y-j) 
              if min_dist > dist: 
                # Get the actual distance using A* search
                act_dist = self.find_actual_distance(newPos,(i,j),successorGameState)
                if min_dist > act_dist:
                  min_dist = act_dist
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos    = successorGameState.getGhostPositions()

        #for one_ghost_state in newGhostStates:
        #  print(one_ghost_state.asdasd)
        #  for one_info in one_ghost_state:
        #    print(one_info)

        newScaredTimes    = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Get the minimum distance from the ghost
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        min_dist_ghost = 999999
        for g_pos in newGhostPositions:
          x,y  = newPos
          dist = abs(x - g_pos[0]) + abs(y - g_pos[1])
          if min_dist_ghost > dist: min_dist_ghost = dist
        if min_dist_ghost > 3: min_dist_ghost = 0
        "*** YOUR CODE HERE ***"
        if min_dist_ghost == 0: min_dist_ghost = 0.002
        g_score = float(successorGameState.getScore())/10.0 + float(1.0 / min_dist) - 3 * float(1.0 / min_dist_ghost)
        #g_score = float(1.0 / min_dist) #- 3 * float(1.0 / min_dist_ghost)
        print(float(successorGameState.getScore())/10.0)

        return g_score

    
    def find_actual_distance(self, curr_pos, des_pos, gameState):
      """
      Use A* search to quickly find the distance
      """
      #x,y = curr_pos
      #next_pos =  [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
      visited_nodes = set()
      fringe_nodes  = util.PriorityQueue()
      fringe_nodes.push(curr_pos,0) 
      visited_nodes.add(curr_pos)
      score_table = {}
      score_table[curr_pos] = 0

      found = False
      while not fringe_nodes.isEmpty():
        # Get the position with minimum score
        pos_eval   = fringe_nodes.pop()
        prev_score = score_table[pos_eval]

        if pos_eval == des_pos:
          found = True
          break
        
        x,y      = pos_eval
        next_pos =  [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        
        for one_pos in next_pos:
          if gameState.hasWall(one_pos[0],one_pos[1]):
            continue
          if one_pos in visited_nodes:
            continue
          manhattanDistance    = abs(one_pos[0] - des_pos[0]) + abs(one_pos[1] - des_pos[1])
          new_score            = prev_score + 1
          score_table[one_pos] = new_score

          visited_nodes.add(one_pos)
          fringe_nodes.push(one_pos,new_score + manhattanDistance)
        
      if not found:
        prev_score = 99999
      return prev_score



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
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

