import copy
import random;
import time as T;

#Require these libraries.
import matplotlib.pyplot as plt 


#Linear Probing.
class HashTable:
    class Entry:
       def __init__(self,k,v):
            self.key = k; self.value = v;

    def __init__(self,m=1):
        if(m <= 0): m = 1;
        self.table = [None]*m;
        self.size = 0;
        self.DELETE = self.Entry(object(),None);
        self.numExistCells = 0;

    def __indexOf(self,key):
        hI = self.__h(key);
        for j in range(len(self.table)):
            e = self.table[hI];
            if(e == None or e.key == key): return hI;
            hI = (hI + 1) % len(self.table);
            
    def __h(self,key):
        key = str(key);
        return (abs(hash(key)))% len(self.table)

    def length(self): return self.size;

    def display(self):
        x,y = [],[];
        for i in range(len(self.table)):
            x.append(i);
            e = self.table[i]
            if(e != None and e != self.DELETE): y.append(1);
            else: y.append(0);
        print("n = ",self.size,"m=",len(self.table),"lambda",self.size/len(self.table));
        plt.bar(x,y,align='edge', width=1);
        plt.show();
    

    def get(self,key):
        hI = self.__indexOf(key);
        e = self.table[hI];
        return None if (e == None or e == self.DELETE) else e.value;

    def put(self,k,v):
        empty = -1;
        oldValue = None;
        hI = self.__h(k);
        
        for j in range(len(self.table)):
            e = self.table[hI]
            if(e == self.DELETE and empty == -1): empty = hI;
            if(e == None or e.key == k): break;
            hI = (hI + 1) % len(self.table);
        
        if(self.table[hI] == None):
            if(empty != -1): hI = empty;
            self.table[hI] = self.Entry(k,v);
            self.size += 1;
            if(empty == -1): self.numExistCells += 1;
            if(self.numExistCells > len(self.table)/2): self.rehash();
            
        elif(self.table[hI].key == k):
            oldValue = self.table[hI].value;
            self.table[hI].value = v;

        return oldValue;

    def remove(self,key):
        hI = self.__indexOf(key);
        if(self.table[hI] != None):
            self.table[hI] = self.DELETE;
            self.size -= 1;

    def rehash(self):
        oldTable = self.table
        self.table = [None] * self.size * 4
        self.size = self.numExistCells = 0;
        for i in range(len(oldTable)):
            entry = oldTable[i]
            if(entry != None and entry != self.DELETE):
                self.put(entry.key,entry.value)
    

class minheap:
    data,s = [],0;
    def __greather(self,i,j):
        return self.data[i].cost() < self.data[j].cost(); #need to defined cost method node.
    def __fix_up(self,n):
        while( n > 0 ):
            p = (n - 1)//2; # parent index;
            if(not self.__greather(n,p)): break; # if parent is more important, it's ending.
            self.data[n],self.data[p] = self.data[p],self.data[n];
            n = p;
    def __fix_down(self,p):
        while(  2*p + 1 < self.s): # if have a left child, do this.
            cI = 2*p + 1;
            if ( cI + 1 < self.s and self.__greather(cI+1,cI)): cI += 1; # if right child is more vital, change index to the right
            if ( self.__greather(p,cI)) : break; #พ่อสำคัญน้อยกว่า ก็ถูกแล้ว
            self.data[p],self.data[cI] = self.data[cI],self.data[p];
            p = cI;
    def size(self): return self.s;
    def peek(self): return None if self.s == 0 else self.data[0];
    def enqueue(self,obj):
        if( obj == None ): return
        self.data.append(obj); self.__fix_up(self.s); self.s += 1;
    def dequeue(self):
        top = self.peek(); self.s -= 1;
        self.data[0] = self.data[self.s]; self.data.pop();
        if(self.s > 0) : self.__fix_down(0);
        return top;



#note status is list of 20 pieces [1,2,3,...,20]

actionList = ['U', 'Ui', 'D', 'Di', 'R', 'Ri', 'L', 'Li', 'B', 'Bi', 'F', 'Fi']


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
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1  

    if action == 'Ui':
        statusAf[6] = statusBf[0]
        statusAf[0] = statusBf[2]
        statusAf[2] = statusBf[4]
        statusAf[4] = statusBf[6]
        
        statusAf[7] = statusBf[1]
        statusAf[1] = statusBf[3]
        statusAf[3] = statusBf[5]
        statusAf[5] = statusBf[7]
        
        if node.lastMove == 'Ui':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1  
            
#-----------------------------------------------------------------------------
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
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1          

    if action == 'Di':     
        statusAf[14] = statusBf[8]
        statusAf[8] = statusBf[10]
        statusAf[10] = statusBf[12]
        statusAf[12] = statusBf[14]
        
        statusAf[15] = statusBf[9]
        statusAf[9] = statusBf[11]
        statusAf[11] = statusBf[13]
        statusAf[13] = statusBf[15]    
        
        if node.lastMove == 'Di':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1  
            
#-----------------------------------------------------------------------------
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
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1          

    if action == 'Ri':  
        statusAf[10] = statusBf[2]
        statusAf[2] = statusBf[4]
        statusAf[4] = statusBf[12]
        statusAf[12] = statusBf[10]
        
        statusAf[17] = statusBf[3]
        statusAf[3] = statusBf[18]
        statusAf[18] = statusBf[11]
        statusAf[11] = statusBf[17]   
        
        if node.lastMove == 'Ri':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1

#-----------------------------------------------------------------------------
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
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1  

    if action == 'Li':       
        statusAf[8] = statusBf[0]
        statusAf[0] = statusBf[6]
        statusAf[6] = statusBf[14]
        statusAf[14] = statusBf[8]
        
        statusAf[16] = statusBf[7]
        statusAf[7] = statusBf[19]
        statusAf[19] = statusBf[15]
        statusAf[15] = statusBf[16]
        
        if node.lastMove == 'Li':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1              
            
#-----------------------------------------------------------------------------
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
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1  
            
    if action == 'Bi':
        statusAf[14] = statusBf[6]
        statusAf[6] = statusBf[4]
        statusAf[4] = statusBf[12]
        statusAf[12] = statusBf[14]
        
        statusAf[19] = statusBf[5]
        statusAf[5] = statusBf[18]
        statusAf[18] = statusBf[13]
        statusAf[13] = statusBf[19]
        
        if node.lastMove == 'Bi':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1  

#-----------------------------------------------------------------------------
    if action == 'F':
        statusAf[0] = statusBf[8]
        statusAf[2] = statusBf[0]
        statusAf[10] = statusBf[2]
        statusAf[8] = statusBf[10]
        
        statusAf[1] = statusBf[16]
        statusAf[17] = statusBf[1]
        statusAf[9] = statusBf[17]
        statusAf[16] = statusBf[9]
        
        if node.lastMove == 'F':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1
            
    if action == 'Fi':
        statusAf[8] = statusBf[0]
        statusAf[0] = statusBf[2]
        statusAf[2] = statusBf[10]
        statusAf[10] = statusBf[8]
        
        statusAf[16] = statusBf[1]
        statusAf[1] = statusBf[17]
        statusAf[17] = statusBf[9]
        statusAf[9] = statusBf[16]
        
        if node.lastMove == 'Fi':
            repMove = node.repMove + 1
            if repMove == 3:
                statusAf = None    
        else:
            repMove = 1

#-----------------------------------------------------------------------------
    child = Node(statusAf, action, repMove, node)
    child.depth = node.depth + 1
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
    target = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
    return node.status == target;
    
def solve1(initial):    
    frontier = [initial]
    solution = []
    visited = [];
    while True:
        if len(frontier) == 0: break
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
                        frontier.append(item);                             
    return(solution,len(visited))

def solve2(initial):    
    frontier = minheap();
    frontier.enqueue(initial);
    solution = []
    h = HashTable(10000);
    while frontier.size() > 0:
        node = frontier.dequeue();
        if goalCheck(node):
            path = [node]
            while node.parent != None : 
                path.insert(0, node.parent)
                node = node.parent
                solution = path
            break
        else:
            for action in actionList:
                new_item = transition(node,action);
                if(new_item == None): continue;
                n = h.get(new_item);
                if(n == None):
                    frontier.enqueue(new_item);
                    h.put(new_item,new_item);
    return(solution,h.length())

class Node():
    depth = 0;
    def __init__(self, status, lastMove, repMove, parent):
        self.status = status
        self.lastMove = lastMove
        self.repMove = repMove
        self.parent = parent

    def cost(self):
        return h_2(self);



def printSolution(solution):
    print("\nOUTPUT\n");
    for node in solution:
        print( node.lastMove,node.status)

def gf(node):
    e1 = [1,3,5,7];  e2 = [1,16,9,17];  e3 = [3,18,11,17];
    e4 = [9,15,13,11]; e5 = [7,19,15,16]; e6 = [5,18,13,19];
    f1 = [0,2,4,6];  f2 = [0,2,8,10];  f3 = [4,6,12,14];
    f4 = [6,0,8,14]; f5 = [2,4,12,10]; f6 = [8,10,12,14];
    mf1 = 0;mf2=0;mf3=0;mf4=0;mf5=0;mf6=0;
    me1=0;me2=0;me3=0;me4=0;me5=0;me6=0;
    for i in f1:
        if(node.status[i] != i+1): mf1 += 1;
    for i in f2:
        if(node.status[i] != i+1): mf2 += 1;
    for i in f3:
        if(node.status[i] != i+1): mf3 += 1;
    for i in f4:
        if(node.status[i] != i+1): mf4 += 1;
    for i in f5:
        if(node.status[i] != i+1): mf5 += 1;
    for i in f6:
        if(node.status[i] != i+1): mf6 += 1;
    for i in e1:
        if(node.status[i] != i+1): me1 += 1;
    for i in e2:
        if(node.status[i] != i+1): me2 += 1;
    for i in e3:
        if(node.status[i] != i+1): me3 += 1;
    for i in e4:
        if(node.status[i] != i+1): me5 += 1;
    for i in e5:
        if(node.status[i] != i+1): me5 += 1;
    for i in e6:
        if(node.status[i] != i+1): me6 += 1;
    mef1 = mf1+me1;mef2 = mf2+me2;
    mef3 = mf3+me3;mef4 = mf4+me4;
    mef5 = mf5 + me5;mef6 = mf6+me6;
    return (max(mef1,mef2,mef3,mef4,mef5,mef6),min(mef1,mef2,mef3,mef4,mef5,mef6))
#!!!!!!!!!!!!!!!!!!!!!!!
W = 2.8;
#!!!!!!!!!!!!!!!!!!!!!!!
#From our experiments.
x = [0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3];

y5 = [0.7404193878173828,0.15450787544250488,0.08415436744689941,
      0.03499746322631836,0.021996498107910156,0.01399850845336914,
      0.010950803756713867,0.00798344612121582,0.008999824523925781,
      0.009994029998779297,0.005002260208129883,0.006072521209716797,
      0.004990816116333008,0.0050048828125,0.0039997100830078125,
      0.004004716873168945,0.0030012130737304688,0.0030133724212646484,
      0.004002571105957031,0.002993345260620117,0.003966331481933594,
      0.003999948501586914,0.003976583480834961,0.003989219665527344,
      0.008011817932128906]

y4 = [10.824617624282837,3.159905195236206,0.6548037528991699,
      0.5505051612854004,0.0530247688293457,0.0350804328918457,
      0.03600192070007324,0.034001827239990234,0.02599930763244629,
      0.011004447937011719,0.012000083923339844,0.009015798568725586,
      0.008016824722290039,0.007993459701538086,0.008999347686767578,
      0.007992744445800781,0.005997896194458008,0.006000041961669922,
      0.004992485046386719,0.006029367446899414,0.005042314529418945,
      0.005978107452392578,0.005002260208129883,0.006028413772583008,
      0.006020307540893555];

y3 = [0.03400301933288574,0.013001441955566406,0.011999130249023438,
      0.012998819351196289,0.02100205421447754,0.010989665985107422,
      0.011992454528808594,0.012998580932617188,0.0119719505310058,
      0.009996175765991211,0.014983177185058594,0.010996341705322266,
      0.011000871658325195,0.01100301742553711,0.0129547119140625,
      0.011004209518432617,0.010035991668701172,0.009998321533203125,
      0.01399374008178711,0.011002302169799805,0.00899648666381836,
      0.01000356674194336,0.00951838493347168,0.009999752044677734,
      0.012501716613769531]

y1 = [0.14000964164733887,0.10901951789855957,0.033005475997924805,0.03199958801269531
     ,0.010998010635375977,0.006998777389526367,0.007998466491699219
     ,0.008002519607543945,0.005999326705932617,0.004996538162231445
     ,0.003998279571533203,0.0039980411529541016,0.003972530364990234
     ,0.0039997100830078125,0.005998134613037109,0.008999109268188477,
      0.008002519607543945,0.008023738861083984,0.007017374038696289,
      0.006998300552368164,0.011001110076904297,0.006970643997192383,
      0.007984399795532227,0.007998466491699219,0.00899195671081543];

y2 = [26.186890840530396,7.351898670196533,1.7020010948181152,
      1.1490402221679688,1.0722637176513672,0.09604120254516602,
      0.09400153160095215,0.08799982070922852,0.06699824333190918,
      0.059035301208496094,0.024042367935180664,0.02199244499206543,
      0.015999794006347656,0.020035505294799805,0.013998985290527344,
      0.016999006271362305,0.010033607482910156,0.010996341705322266,
      0.009015321731567383,0.012004852294921875,0.008989810943603516,
      0.013995170593261719,0.008004426956176758,0.010004281997680664,
      0.011002540588378906]

y = [(y1[i]+y2[i]+y3[i]+y4[i]+y5[i])/5 for i in range(len(x))];

def h_0(node): 
    correct = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
    c = 0;
    for i in range(len(correct)):
        if(node.status[i] != correct[i]): c+= 1;
    return c +node.depth*W;

def h_1(node): 
    corner = [1, 3, 5, 7, 9, 11, 13 ,15]
    Index_corner = [0, 2, 4, 6, 8, 10, 12, 14]
    n = 0
    for i in range(8) :
        if node.status[Index_corner[i]] != corner[i]:
            n += 1
    return(n +node.depth*W) 

def h_2(node): # Test for this function
    edge = [2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20]
    Index_edge = [1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19]
    n = 0
    for i in range(12) :
        if node.status[Index_edge[i]] != edge[i]:
            n += 1
    return(n) + node.depth*W

def h_3(node):
    n = h_1(node) + h_2(node)
    return(n)+node.depth*W 

def h_4(node):
    m = min(h_1(node), h_2(node))
    return m+node.depth*W; 

def h_5(node):
    f1 = [0,2,4,6];  f2 = [0,2,8,10];  f3 = [4,6,12,14];
    f4 = [6,0,8,14]; f5 = [2,4,12,10]; f6 = [8,10,12,14];
    m1 = 0;m2 = 0;m3 = 0;m4 = 0;m5=0;m6 =0;
    for i in f1:
        if(node.status[i] != i+1): m1 += 1;
    for i in f2:
        if(node.status[i] != i+1): m2 += 1;
    for i in f3:
        if(node.status[i] != i+1): m3 += 1;
    for i in f4:
        if(node.status[i] != i+1): m4 += 1;
    for i in f5:
        if(node.status[i] != i+1): m5 += 1;
    for i in f6:
        if(node.status[i] != i+1): m6 += 1;
    return max(m1,m2,m3,m4,m5,m6) +node.depth*W; 

def h_6(node): 
    e1 = [1,3,5,7];  e2 = [1,16,9,17];  e3 = [3,18,11,17];
    e4 = [9,15,13,11]; e5 = [7,19,15,16]; e6 = [5,18,13,19];
    m1 = 0;m2 = 0;m3 = 0;m4 = 0;m5=0;m6 =0;
    for i in e1:
        if(node.status[i] != i+1): m1 += 1;
    for i in e2:
        if(node.status[i] != i+1): m2 += 1;
    for i in e3:
        if(node.status[i] != i+1): m3 += 1;
    for i in e4:
        if(node.status[i] != i+1): m4 += 1;
    for i in e5:
        if(node.status[i] != i+1): m5 += 1;
    for i in e6:
        if(node.status[i] != i+1): m6 += 1;

    #print(m1,m2,m3,m4,m5,m6);
    return max(m1,m2,m3,m4,m5,m6)+node.depth*W;
    #return m1+m2+m3+m4+m5+m6


def h_7(node): 
    return gf(node)[0] + gf(node)[1] +node.depth*W;

    
def h_8(node):
    return h_7(node) + h_6(node) +node.depth*W; 




#generating
SMILE = Node([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], None, None, None)
for i in range(4):
    act = random.choice(actionList)
    SMILE = transition(SMILE, act)
print("INPUT\n");

S2 = Node([9, 2, 3, 4, 5, 6, 7, 17, 15, 12, 13, 14, 1, 8, 11, 20, 16, 18, 19, 10],None,None,None);
print("initail state",S2.status,"h:",S2.cost());
for act in actionList:
    sample = transition(S2, act)
    print(act,sample.status,"h:",sample.cost())

n = SMILE.status
#5 samples for average case.
#initial = Node([11, 18, 3, 6, 15, 20, 7, 2, 13, 19, 5, 14, 9, 17, 1, 10, 8, 12, 4, 16] ,None,None,None)
#initial = Node([7, 8, 13, 2, 9, 4, 5, 6, 3, 12, 15, 19, 1, 16, 11, 18, 10, 14, 17, 20],None,None,None);
#initial = Node([9, 17, 3, 12, 11, 8, 1, 2, 7, 19, 13, 6, 5, 14, 15, 16, 10, 4, 18, 20],None,None,None);
#initial = Node([5, 19, 13, 6, 3, 12, 7, 2, 15, 16, 9, 10, 11, 20, 1, 14, 17, 4, 18, 8],None,None,None);
#initial = Node([1, 8, 3, 6, 15, 20, 7, 2, 5, 12, 13, 14, 9, 17, 11, 18, 10, 4, 19, 16],None,None,None);

initial = Node(n,None,None,None);
print(initial.status);
def exp():
    print("\n Bread first search is searching ... ");
    start = T.time();
    sln1,n1 = solve1(initial)
    print("\ntime : ",T.time() - start,"s");
    printSolution(sln1)
    print("Bread frist search is OK.","Explores :",n1,"nodes\n");

def exp1():
    print("\nHeuristic search is searching ...  W = ",W);
    start = T.time();
    sln2,n2 = solve2(initial);
    print("\ntime : ",T.time() - start,"s");
    printSolution(sln2);
    print("Heuristic search is OK.","Explores :",n2,"nodes\n");
    
plt.figure(figsize=(8,7));
plt.scatter(x,y);
plt.title(label = " Relation between different weights and diffetent running time from selected input");
plt.ylabel("Time complexity (seconds)",fontsize=13);
plt.xlabel("weights",fontsize=13);
good = min(y); gI = y.index(good);gx = x[gI];
print("good weight is :",gx);
plt.show();
#exp();
exp1();
#print("Best weight is ",gx);



# ค่านี้เกิดจากรันทีละรอบละรอบ ทั้งหมด 50 รอบ
testcases = [0.0019960403442382812,0.0059969425201416016,0.0069980621337890625,
             0.003998517990112305,0.0019998550415039062,0.011001825332641602,
             0.00897359848022461,0.028029203414916992,0.008004188537597656,
             0.0049762725830078125,0.04501771926879883,0.004003286361694336,
             0.006022930145263672,0.021980762481689453,0.004998445510864258,
             0.025992393493652344,0.0010132789611816406,0.012520790100097656,
             0.018994808197021484,0.0010013580322265625,0.0029981136322021484,
             0.0009889602661132812,0.004998683929443359,0.00398564338684082,
             0.001993894577026367,0.008989334106445312,0.0009620189666748047,
             0.03700613975524902,0.007012844085693359,0.02603888511657715,
             0.012000083923339844,0.00800633430480957,0.019033432006835938,
             0.020998239517211914,0.021759748458862305,0.0009953975677490234,
             0.004988908767700195,0.044000864028930664,0.0009961128234863281,
             0.007001638412475586,0.004996776580810547,0.017995119094848633,
             0.015038251876831055,0.012067079544067383,0.005954742431640625,
             0.011020421981811523,0.008013248443603516,0.02003645896911621,
             0.008043050765991211,0.007996320724487305,0.012999773025512695,
             0.006993293762207031,0.010015726089477539,0.00802159309387207,
             0.006000995635986328,0.005287647247314453,0.023990392684936523,
             0.009000062942504883,0.0070037841796875,0.0019931793212890625];
    

plt.bar([x for x in range(60)],testcases,align="center",width = 0.7);
plt.ylabel("Time complexity (seconds)",fontsize=13);
plt.title("Others 30 different inputs",fontsize=13);
print("avg test cases time complexity",sum(testcases)/60,"seconds");
plt.show();








    



    
