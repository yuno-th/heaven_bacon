import copy

#note status is list of 20 pieces [1,2,3,...,20]
class Node():
    def __init__(self, status, lastMove, repMove, parent):
        self.status = status
        self.lastMove = lastMove
        self.repMove = repMove
        self.parent = parent

actionList = ['U', 'D', 'R', 'L', 'B', 'F']

def transition(node, action):
    statusBf = copy.deepcopy(node.status)
    statusAf = copy.deepcopy(node.status)
    
    if action == 'U':
        statusAf[0] = statusBf[6]
        statusAf[2] = statusBf[0]
        statusAf[4] = statusBf[2]
        statusAf[6] = statusBf[4]
        
        statusAf[1] = statusBf[7]
        statusAf[3] = statusBf[1]
        statusAf[5] = statusBf[3]
        statusAf[7] = statusBf[5]   
        
        if node.lastMove == 'U':
            repMove = node.repMove + 1
            if repMove == 4:
                statusAf = None    
        else:
            repMove = 1  

    if action == 'D':     
        statusAf[8] = statusBf[14]
        statusAf[10] = statusBf[8]
        statusAf[12] = statusBf[10]
        statusAf[14] = statusBf[12]
        
        statusAf[9] = statusBf[15]
        statusAf[11] = statusBf[9]
        statusAf[13] = statusBf[11]
        statusAf[15] = statusBf[13]    
        
        if node.lastMove == 'D':
            repMove = node.repMove + 1
            if repMove == 4:
                statusAf = None    
        else:
            repMove = 1          

    if action == 'R':  
        statusAf[2] = statusBf[10]
        statusAf[4] = statusBf[2]
        statusAf[12] = statusBf[4]
        statusAf[10] = statusBf[12]
        
        statusAf[3] = statusBf[17]
        statusAf[18] = statusBf[3]
        statusAf[11] = statusBf[18]
        statusAf[17] = statusBf[11]   
        
        if node.lastMove == 'R':
            repMove = node.repMove + 1
            if repMove == 4:
                statusAf = None    
        else:
            repMove = 1          

    if action == 'L':       
        statusAf[0] = statusBf[8]
        statusAf[6] = statusBf[0]
        statusAf[14] = statusBf[6]
        statusAf[8] = statusBf[14]
        
        statusAf[7] = statusBf[16]
        statusAf[19] = statusBf[7]
        statusAf[15] = statusBf[19]
        statusAf[16] = statusBf[15]
        
        if node.lastMove == 'L':
            repMove = node.repMove + 1
            if repMove == 4:
                statusAf = None    
        else:
            repMove = 1  

    if action == 'B':
        statusAf[6] = statusBf[14]
        statusAf[4] = statusBf[6]
        statusAf[12] = statusBf[4]
        statusAf[14] = statusBf[12]
        
        statusAf[5] = statusBf[19]
        statusAf[18] = statusBf[5]
        statusAf[13] = statusBf[18]
        statusAf[19] = statusBf[13]
        
        if node.lastMove == 'B':
            repMove = node.repMove + 1
            if repMove == 4:
                statusAf = None    
        else:
            repMove = 1  

    if action == 'F':
        statusAf[0] = statusBf[8]
        statusAf[2] = statusBf[0]
        statusAf[10] = statusBf[2]
        statusAf[8] = statusBf[10]
        
        statusAf[1] = statusBf[16]
        statusAf[17] = statusBf[1]
        statusAf[10] = statusBf[17]
        statusAf[16] = statusBf[10]
        
        if node.lastMove == 'F':
            repMove = node.repMove + 1
            if repMove == 4:
                statusAf = None    
        else:
            repMove = 1  
    
    child = Node(statusAf, action, repMove, node)
    if statusAf == None :
        return(None)
    else:
        return(child)
    
def expand(node):
    listNextNode = []
    for action in actionList:
        child = transition(node, action)
        if child != None:
            listNextNode.append(child)
    return(listNextNode)

def goalCheck(node):
    if node.status == [1,2,3,4,5,6,7,8,9,10,
                       11,12,13,14,15,16,17,18,19,20]:
        return(True)
    else:
        return(False)
    
def solve(initial):    
    frontier = [initial]
    solution = []
    visited = []
    while True:
        if len(frontier) == 0:
            break
        else:
            node = frontier.pop(0) #HERE!!!!!!!!!!!!!!
            if goalCheck(node):
                path = [node]
                while node.parent != None : 
                    path.insert(0, node.parent)
                    node = node.parent                
                solution = path
                break
            else:
                for item in expand(node):
                    repeat = False
                    for oldNode in visited:
                        if oldNode.status == item.status :
                            repeat = True
                            break
                    if not repeat:
                        visited.append(item)
                        frontier.append(item)
                #print(len(visited))                              
    return(solution)

def printSolution(solution):
    for node in solution:
        print( node.lastMove,node.status)

def h_1(node):
    corner = [1, 3, 5, 7, 9, 11, 13 ,15]
    Index_corner = [0, 2, 4, 6, 8, 10, 12, 14]
    n = 0
    for i in range(8) :
        if node.status[Index_corner[i]] != corner[i]:
            n += 1
    return(n)


def h_2(node):
    edge = [2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20]
    Index_edge = [1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19]
    n = 0
    for i in range(12) :
        if node.status[Index_edge[i]] != edge[i]:
            n += 1
    return(n)

def h_3(node):
    n = h_1(node) + h_2(node)
    return(n)

def h_4(node):
    m = min(h_1(node), h_2(node))

#initial = Node([3,4,5,6,7,8,1,2,9,10,11,12,13,14,15,16,17,18,19,20], None, None, None)
#initial = Node([9,8,13,6,4,11,7,1,2,3,5,10,12,14,15,16,19,17,18,20], None, None, None)
initial = Node([7, 18, 18, 6, 16, 2, 14, 12, 19, 10, 9, 5, 11, 4, 18, 8, 20, 17, 11, 14],None,None,None)
sln = solve(initial)
printSolution(sln)
