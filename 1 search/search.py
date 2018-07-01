# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH

    return  [s, s, n, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    visited_nodes = []
    start_node    = problem.getStartState()
    visited_nodes.append(start_node)
    curr_node     = start_node
    q             = util.Queue()
    directions    = util.Queue()
    q.push(curr_node)
    goal_found    = problem.isGoalState(curr_node)

    while not goal_found:
        nxt_node_list = problem.getSuccessors(curr_node)
        nxt_node_found = False

        # Check if a child can be found which has not been visited
        for node in nxt_node_list:
            nxt_node  = node[0]
            move      = node[1]
            if nxt_node not in visited_nodes:
                nxt_node_found = True        # mark that a child node has been found
                q.push(nxt_node)             # add the node in the tree
                directions.push(move)        # add the direction
                visited_nodes.append(nxt_node) # mark the node as visited
                break

        # If child not found, go to parent
        if not nxt_node_found:
            q.list.pop(0)
            directions.list.pop(0)

        if q.isEmpty(): break

        curr_node   = q.list[0]
        goal_found  = problem.isGoalState(curr_node)

    final_moves = []
    while not directions.isEmpty():
        final_moves.append(directions.pop())
    
    return final_moves
    #util.raiseNotDefined()

#def breadthFirstSearch(problem):
#    """Search the shallowest nodes in the search tree first."""
#    "*** YOUR CODE HERE ***"
#    # Keep track of visited nodes
#    #visited_nodes = set()
#
#    # Get the start node
#    start_node    = problem.getStartState()
#
#    # Mark the start node as visited
#    #visited_nodes.add(start_node)
#
#    curr_node     = start_node
#
#    goal_found    = problem.isGoalState(curr_node)
#    fringe_node   = set()
#    fringe_node.add(start_node)
#
#    parent  = {}
#    my_move = {}
#    parent[start_node]  = None
#    my_move[start_node] = None
#
#    while not goal_found:
#        # Keep list of next nodes to explore
#        nxt_nodes_to_explore = []
#        # Explore child of current level
#        for one_node in curr_nodes_to_explore:
#            for nxt_node in problem.getSuccessors(one_node):
#                pos = nxt_node[0]
#                mv  = nxt_node[1]
#                #if pos not in visited_nodes:
#                nxt_nodes_to_explore.append(pos)
#                visited_nodes.add(pos)
#                parent[pos] = one_node
#                my_move[pos] = mv
#                goal_found = problem.isGoalState(pos)
#                #final_pos = pos
#                if goal_found:
#                    final_pos = pos
#                    break
#
#            if goal_found:
#                break
#        if goal_found:
#            break
#        curr_nodes_to_explore = nxt_nodes_to_explore
#    
#    print(final_pos)
#    cur_node = final_pos
#    cur_mv   = my_move[cur_node]
#    q        = util.Queue()
#
#    while cur_mv is not None:
#        q.push(cur_mv)
#        cur_node = parent[cur_node]
#        cur_mv   = my_move[cur_node]
#
#    print(q.list)
#    return q.list
#                    
#
#    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Get the start node
    start_state    = problem.getStartState()
    print(start_state)

    # Define a stack
    plan_stack    = util.Queue()
    start_plan    = [start_state]   # node, cost
    plan_stack.push(start_plan)

    # Visited nodes
    visited_nodes = set(start_state)

    goal_found = False

    while not goal_found:
        # Get the plan from the stack
        plan_to_expand = plan_stack.pop()
        node_to_exp    = plan_to_expand[-1]
        all_nxt_nodes  = problem.getSuccessors(node_to_exp)

        # Traverse through all the next nodes
        for nxt_node in all_nxt_nodes:
            nxt_pos  = nxt_node[0]

            if nxt_pos in visited_nodes:                   # Check if node is already visited
                continue

            visited_nodes.add(nxt_pos)                     # Add the node to visited nodes
            nxt_plan = plan_to_expand + [nxt_pos]          # add node to the plan
            plan_stack.push(nxt_plan)                      # push the plan into the stack
            goal_found = problem.isGoalState(nxt_pos)      # Check if goal is achieved
            if goal_found:
                break
        
    
    print(goal_found)
    print(nxt_plan)

    moves = []
    # Convert plan to moves
    for i in range(len(nxt_plan) - 1):
        for nxt_node in problem.getSuccessors(nxt_plan[i]):
            nxt_pos = nxt_node[0]
            nxt_mv  = nxt_node[1]
            if nxt_pos == nxt_plan[i+1]:
                moves.append(nxt_mv)
                break
    
    return moves

            

        # Calculate the minimum plan cost    
        #min_val = float("inf")
        #for one_plan in plan_stack:
        #    plan_cost = one_plan[1]
        #    if plan_cost < min_val:
        #        min_val = plan_cost

        ## Expand the nodes with minimum plan cost
        #for one_plan in plan_stack:
        #    plan_cost = one_plan[1]
        #    if plan_cost == min_val:
        #        plan_step = one_plan[0]    
        #        # Expand the last node of plan
        #        last_node = plan_step[end]
        #        for nxt_node in problem.getSuccessors(last_node):



    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    path_to_point = {}
    cost_to_point = {}

    # Get the start node
    start_node   = problem.getStartState()
    fringe_node  = [start_node]
    path_to_point[start_node] = []
    cost_to_point[start_node] = problem.getCostOfActions(path_to_point[start_node])

    goal_found = False

    while(not goal_found):
    #for i in range(100):    
        nodes_to_expand = set()
        # get max value node in the fringe node
        min_val = float("inf")
        for one_node in fringe_node:
            # Compute the cost to reach a node
            if cost_to_point[one_node] < min_val:
                min_val = cost_to_point[one_node]
        
        for one_node in fringe_node:
            # Compute the cost to reach a node
            if cost_to_point[one_node] == min_val:
                nodes_to_expand.add(one_node)
                fringe_node.remove(one_node)

        # Expand the fringe node 
        for one_node in nodes_to_expand:
            path_to_parent = path_to_point[one_node]
            for nxt_node in problem.getSuccessors(one_node):
                pos = nxt_node[0]
                mv  = nxt_node[1]
                # check if point already present in path to point
                prev_cost = float("inf")
                if pos in cost_to_point:
                    prev_cost = cost_to_point[pos]
                new_path = path_to_parent + [mv]
                if prev_cost > problem.getCostOfActions(new_path):
                    path_to_point[pos] = new_path
                    cost_to_point[pos] = problem.getCostOfActions(new_path)
                    fringe_node.append(pos)

        # Check if destination is reached in the fringe node
        for one_node in fringe_node:
            if problem.isGoalState(one_node):
                final_node = one_node
                goal_found = True
                break
        
        #print(len(fringe_node))
    print(final_node)
    print(path_to_point[final_node])
    return path_to_point[final_node] 

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize data structures
    parent_node    = {}
    path_to_node   = {}
    priority_queue = util.PriorityQueue()

    p_c = 0.5
    h_c = 1 - p_c

    # Get the start node
    start_node               = problem.getStartState()
    parent_node[start_node]  = None
    path_to_node[start_node] = []
    priority_queue.update(start_node, 0)

    #goal_found = False

    while not priority_queue.isEmpty():
        # Get the next node
        node_to_expand = priority_queue.pop()
        # Check if goal state is reached
        if problem.isGoalState(node_to_expand):
            break
        next_nodes     = problem.getSuccessors(node_to_expand)
        path_to_parent = path_to_node[node_to_expand]

        for one_node in next_nodes:
            point, move, cost = one_node
            curr_path         = path_to_node[node_to_expand] + [move]
            curr_cost         = problem.getCostOfActions(curr_path)
            heuristic_cost    = heuristic(point, problem)
            # Check if current node already exists in the previously visited nodes
            if point in path_to_node:
                prev_cost = problem.getCostOfActions(path_to_node[point])
                if prev_cost > curr_cost:
                    path_to_node[point] = curr_path
                    priority_queue.update(point, curr_cost + heuristic_cost)
            
            else:
                path_to_node[point] = curr_path
                priority_queue.update(point, curr_cost + heuristic_cost)
            
            #    current_cost = problem.getCostOfActions(point) * p_c + heuristic(point, problem) * h_c

    print(node_to_expand)    
    return path_to_node[node_to_expand]
            
#        nodes_to_expand = set()
#        # get max value node in the fringe node
#        min_val = float("inf")
#        for one_node in fringe_node:
#            # Compute the cost to reach a node
#            total_cost = cost_to_point[one_node] * p_c + heuristic(one_node,problem) * h_c
#            if total_cost < min_val:
#                min_val = total_cost
#        
#        for one_node in fringe_node:
#            # Compute the cost to reach a node
#            total_cost = cost_to_point[one_node] * p_c + heuristic(one_node,problem) * h_c
#            if total_cost == min_val:
#                nodes_to_expand.add(one_node)
#                fringe_node.remove(one_node)
#
#        # Expand the fringe node 
#        for one_node in nodes_to_expand:
#            path_to_parent = path_to_point[one_node]
#            for nxt_node in problem.getSuccessors(one_node):
#                pos = nxt_node[0]
#                mv  = nxt_node[1]
#                # check if point already present in path to point
#                prev_cost = float("inf")
#                if pos in cost_to_point:
#                    prev_cost = cost_to_point[pos]
#                new_path = path_to_parent + [mv]
#                if prev_cost > problem.getCostOfActions(new_path):
#                    path_to_point[pos] = new_path
#                    cost_to_point[pos] = problem.getCostOfActions(new_path)
#                    fringe_node.append(pos)
#
#        # Check if destination is reached in the fringe node
#        for one_node in fringe_node:
#            if problem.isGoalState(one_node):
#                final_node = one_node
#                goal_found = True
#                break
#        
#        #print(len(fringe_node))
#    print(final_node)
#    print(path_to_point[final_node])
#    return path_to_point[final_node] 

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
