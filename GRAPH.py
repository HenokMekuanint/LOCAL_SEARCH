from ast import JoinedStr
import sys
import os
from collections import deque
from typing import List 
from numpy import Infinity
from timeit import default_timer as timer
import random
import pandas as pd
from matplotlib import markers, pyplot as plt 

import string

heurstic={
            "Oradea" :[47.0465005,21.9189438],
            "Zerind" :[46.622511 ,21.517419],
            "Arad" :[46.166667 ,21.316667],
            "Timisoara" :[45.759722 ,21.23],
            "Lugoj" :[45.68861 ,21.90306],
            "Mehadia" :[44.904114 ,22.364516],
            "Drobeta" :[44.636923 ,22.659734],
            "Craiova" :[44.333333 ,23.816667000000052],
            "Sibiu" :[45.792784 ,24.152068999999983],
            "RimnicuVilcea" :[45.099675 ,24.369318],
            "Fagaras" :[45.8416403 ,24.9730954],
            "Giurgiu" :[43.9037076 ,25.9699265],
            "Bucharest" :[44.439663 ,26.096306],
            "Urziceni" :[44.7165317 ,26.641121],
            "Eforie" :[44.058422 ,28.633607],
            "Hirsova" :[44.6833333 ,27.9333333],
            "Vaslui" :[46.640692 ,27.727647],
            "Iasi":[47.156944 ,27.590278000000012],
            "Neamt" : [47.2 ,26.3666667],
            "Pitesti":[44.856480,24.869182],
        }
final_bfs_time=[]
final_bfs_sol_len=[]







graph={}
class dequeue:
    def __init__(self):
        self.dequeue=[]
    def appendleft(self,variable):
        self.dequeue.insert(0,variable)
class stack:
    def __init__(self):
        self.stack=[]
    def push(self,variable):
        self.stack.append(variable)
    def pop(self):
        self.stack.pop(-1)
    def size(self):
        return len(self.stack)
class queue:
    def __init__(self):
        self.queue=[]
        
    def push(self,variable):
        self.queue.append(variable)
    def pop(self):
        return (self.queue.pop(0))
    def size(self):
        return len(self.queue)

class edges:
    def __init__(self,CONNECTED_EDGES,WEIGHT):
        self.CONNECTED_EDGES=CONNECTED_EDGES
        self.WEIGHT=WEIGHT
    def __repr__(self):
        return "(" + str(self.CONNECTED_EDGES) + "," + str(self.WEIGHT) + ")"


class graph:
    def __init__(self):
        self.Huristic_for_generated_graph={}
        self.adjacent={}
        self.generated_graph={}
        self.file=open(os.path.join(sys.path[0], "graphdata.txt"), "r")
        read=self.file.readlines()
        self.modified=[]
        self.newmodified=[]
        for line in read:
            if line[-1]=='\n':
                self.modified.append(line[:-1])
        for i in self.modified:
            str=i
            new=str.split()
            for j in new:
                if new.index(j)==2:
                    new[2]=int(j)
            self.newmodified.append(new)
        for i in range(0,len(self.newmodified)):
            self.add_adjacent_nodes(self.newmodified[i][0],self.newmodified[i][1],self.newmodified[i][2])
    def add_adjacent_nodes(self,node_1,node_2,weight):
        if node_1 not in self.adjacent:
            self.adjacent[node_1]=[]
        if node_2 not in self.adjacent:
            self.adjacent[node_2]=[]
        edge1=edges(node_2,weight)
        self.adjacent[node_1].append(edge1)
        edge2=edges(node_1,weight)
        self.adjacent[node_2].append(edge2)
        
    def print_original_graph(self):
        for i in self.adjacent:
            print(i,"->",self.adjacent[i])

    def print_generated_graph(self) :
        for i in self.generated_graph:
            print(i,"->",self.generated_graph[i])
class search:
    def __init__ (self):
        pass
    def BREAD_FIRST_SEARCH(self,graph,start,end, performance=True):
        if start not in graph.adjacent or end not in graph.adjacent:
            return "there is no such kind of node"
        visited_nodes={}
        que=queue()
        que.push(start)
        there_is_possible_path=False
        visited_nodes[start]=True
        PATH_TRACER_TO_RETRACE={}
        PATH_TRACER_TO_RETRACE[start]=None
        while(que.size!=0):
            CURRENT_POSITION=que.pop()
            if CURRENT_POSITION==end:
                there_is_possible_path=True
                break
            
            for neighbour in graph.adjacent[CURRENT_POSITION]:
                NEXT_STEP=neighbour.CONNECTED_EDGES
                if NEXT_STEP not in visited_nodes:
                    que.push(NEXT_STEP)
                    PATH_TRACER_TO_RETRACE[NEXT_STEP]=CURRENT_POSITION
                    visited_nodes[NEXT_STEP]=True
        path=[]
        if there_is_possible_path:
            path.append(end)
            while PATH_TRACER_TO_RETRACE[end] is not None:
                path.append(PATH_TRACER_TO_RETRACE[end]) 
                end = PATH_TRACER_TO_RETRACE[end]
            path.reverse()
            Joiner=" -> "
            time=None
            if performance:
                time=self.Average_time_and_length_calculator_for_bfs(graph)
                if type(time)==list:
                    return[[Joiner.join(path)],"Average time in micro sec ->",float(("{:.4f}".format(time[0]))),"Average path length->",float(("{:.4f}".format(time[1])))]
            return [path,len(path)]
        else:
            return "there is no possible path"


    def DEPTH_FIRST_SEARCH_ITERATIVE_SOLUTION(self,graph,start,end,performance=True):
        if start not in graph.adjacent or end not in graph.adjacent:
            return "there is no such kind of node"
        stack = []                                                                                                      
        visited = set()
        path = []
        root = start
        stack.append(root)
        visited.add(root)
        
        while stack:
            parent = stack.pop()
            path.append(parent)
            if parent == end: break
            
            for CHILDREN in graph.adjacent[parent]:
                nextnode=CHILDREN.CONNECTED_EDGES
                if nextnode not in visited:
                    visited.add(nextnode)
                    stack.append(nextnode)
        Joiner=" -> "
        if performance:
            time=self.Average_time_and_length_calculator_for_dfs(graph)
            if type(time)==list:
                    return[[Joiner.join(path)],"Average time in micro sec ->",float(("{:.4f}".format(time[0]))),"Average path length->",float(("{:.4f}".format(time[1])))]
        return [Joiner.join(path),len(path)]


    def DJKSTRA_SHORTEST_PATH(self,graph,start,end,performance=True):
        if start not in graph.adjacent or end not in graph.adjacent:
            return "there is no such kind of node"
        previous_node = {}
        visited = graph.adjacent.copy() 

        distance_counter_from_the_start = {}
        for node in graph.adjacent:
            if node==start:
                distance_counter_from_the_start[node]=0
            else:
                distance_counter_from_the_start[node]=Infinity
        for node in graph.adjacent:
            previous_node[node]=None

        while visited:
            current_position = min(visited, key=lambda node: distance_counter_from_the_start[node])
            visited.pop(current_position)

            if distance_counter_from_the_start[current_position] == Infinity:
                break
            for neighbor in graph.adjacent[current_position]:
                new_path = distance_counter_from_the_start[current_position] + int(neighbor.WEIGHT)
                if new_path < distance_counter_from_the_start[neighbor.CONNECTED_EDGES]:
                    distance_counter_from_the_start[neighbor.CONNECTED_EDGES] = new_path
                    previous_node[neighbor.CONNECTED_EDGES] = current_position
            if current_position == end:
                break 
        path = deque()
        current_position = end
        while previous_node[current_position] is not None:
            path.appendleft(current_position)
            current_position = previous_node[current_position]
        path.appendleft(start)
        mylist=list()
        
        Joiner=" -> "
        
        if performance:
            time=self.Average_time_and_length_calculator_for_djkstra(graph)
            if type(time)==list:
                return[[Joiner.join(path)],"Average time in micro sec ->",float(("{:.4f}".format(time[0]))),"Average path length->",float("{:.4f}".format(time[1]))]
        return [Joiner.join(path),len(path)]

        
    def A_star_algorithm(self,graph,start, end,performance=True):
        if start not in graph.adjacent or end not in graph.adjacent:
            return "there is no such kind of node"
        PATH_TO_BE_TRACED =[]
        ADDITION_OF_PATHS = []
        PATH_TO_BE_TRACED.append(start)
        GRAPH = {}               
        PREVIOUS_NODES = {}         
        GRAPH[start] = 0
        PREVIOUS_NODES[start] = start
        while len(PATH_TO_BE_TRACED) > 0:
            n = None
            for v in PATH_TO_BE_TRACED:
                if n == None or  GRAPH[v] + self.HEURISTIC_VALUE_FOR_THE_CITIES(v, end) < GRAPH[n] + self.HEURISTIC_VALUE_FOR_THE_CITIES(n, end):
                    n = v
            if n == end or graph.adjacent[n]== None:
                pass
            else:
                for edge in graph.adjacent[n]:
                    NEIGHBOUR=edge.CONNECTED_EDGES
                    WEIGHT=edge.WEIGHT
                    if NEIGHBOUR not in PATH_TO_BE_TRACED and NEIGHBOUR not in ADDITION_OF_PATHS:
                        PATH_TO_BE_TRACED.append(NEIGHBOUR)
                        PREVIOUS_NODES[NEIGHBOUR] = n
                        GRAPH[NEIGHBOUR] = GRAPH[n] + WEIGHT
                    else:
                        if GRAPH[NEIGHBOUR] > GRAPH[n] + WEIGHT:
                            GRAPH[NEIGHBOUR] = GRAPH[n] + WEIGHT
                            PREVIOUS_NODES[NEIGHBOUR] = n
                            if NEIGHBOUR in ADDITION_OF_PATHS:
                                ADDITION_OF_PATHS.pop(NEIGHBOUR)
                                PATH_TO_BE_TRACED.append(NEIGHBOUR)
            if n == None:
                print('Path does not exist!')
                return None
            path = []

            if n == end:
                while PREVIOUS_NODES[n] != n:
                    path.append(n)
                    n = PREVIOUS_NODES[n]
                path.append(start)
                path.reverse()
                if performance:
                    time=self.performance_for_A_Star(graph)
                    if type(time)==list:
                        return[[" -> ".join(path)],"Average time in micro sec ->",time[0],"Average path length->",time[1]]
                return path,len(path)

            PATH_TO_BE_TRACED.remove(n)
            ADDITION_OF_PATHS.append(n)
        
    def HEURISTIC_VALUE_FOR_THE_CITIES(self,start,end):
        start_coordinate = heurstic[start]
        end_coordinate = heurstic[end]
        import math
        return math.sqrt(
            (end_coordinate[1]-start_coordinate[1])**2+
            (end_coordinate[0]-start_coordinate[0])**2
        )
    def Average_time_and_length_calculator_for_bfs(self,graph):
        average_time_for_bfs=[]
        average_length_for_bfs=[]
        time_array_breadth=[]
        solution_length_array_breadth=[]
        for i in graph.adjacent:
            for j in graph.adjacent:
                if i==j:
                    continue
                else:
                    new_Start=timer()
                    returned=self.BREAD_FIRST_SEARCH(graph,i,j, performance=False)
                    new_end=timer()
                    time_array_breadth.append(float(("{:.6f}".format(new_end-new_Start)))*1000000)
                    solution_length_array_breadth.append(returned[-1])
            average_time_for_bfs.append(sum(time_array_breadth)/20) 
            average_length_for_bfs.append(sum(solution_length_array_breadth)/20)

        return [sum(average_time_for_bfs)/20,sum(average_length_for_bfs)/20]
    def Average_time_and_length_calculator_for_dfs(self,graph):
        average_time_for_dfs=[]
        average_length_for_dfs=[]
        time_array_depth=[]
        solution_length_array_depth=[]
        for i in graph.adjacent:
            for j in graph.adjacent:
                if i==j:
                    continue
                else:
                    new_Start=timer()
                    returned=self.DEPTH_FIRST_SEARCH_ITERATIVE_SOLUTION(graph,i,j, performance=False)
                    new_end=timer()
                    time_array_depth.append(float(("{:.6f}".format(new_end-new_Start)))*1000000)
                    solution_length_array_depth.append(returned[-1])
            average_time_for_dfs.append(sum(time_array_depth)/20) 
            average_length_for_dfs.append(sum(solution_length_array_depth)/20)
        return [sum( average_time_for_dfs)/20,sum(average_length_for_dfs)/20]
            
    def Average_time_and_length_calculator_for_djkstra(self,graph):
        average_time_for_dkstra=[]
        average_length_for_dkstra=[]
        time_array_dkstra=[]
        solution_length_array_dkstra=[]
        for i in graph.adjacent:
            for j in graph.adjacent:
                if i==j:
                    continue
                else:
                    start_time=timer()
                    returned=self.DJKSTRA_SHORTEST_PATH(graph,i,j,performance=False)
                    end_time=timer()
                    time_array_dkstra.append(float(("{:.6f}".format(end_time-start_time)))*1000000)
                    solution_length_array_dkstra.append(returned[-1])
            average_time_for_dkstra.append(sum(time_array_dkstra)/20) 
            average_length_for_dkstra.append(sum(solution_length_array_dkstra)/20)
        return [sum( average_time_for_dkstra)/20,sum( average_length_for_dkstra)/20]
    def performance_for_A_Star(self,graph):
        average_time_for_astar=[]
        average_length_for_astar=[]
        time_array_astar=[]
        solution_length_array_astar=[]
        for i in graph.adjacent:
            for j in graph.adjacent:
                if i==j:
                    continue
                else:
                    start_time=timer()
                    returned=self.A_star_algorithm(graph,i,j,performance=False)
                    end_time=timer()
                    time_array_astar.append(float(("{:.6f}".format(end_time-start_time)))*100000)
                    solution_length_array_astar.append(returned[-1])
            average_time_for_astar.append(sum(time_array_astar)/20)
            average_length_for_astar.append(sum(solution_length_array_astar)/20)
            
        return [sum(average_time_for_astar)/20,sum(average_length_for_astar)/20]
    
    def generate_string_and_connect_randomly(self,graph):
        array=[]
        letters = string.ascii_uppercase
        node_2=''.join(random.choice(letters) for i in range(10))
        max_connection_of_node=random.randint(2,4)
        if node_2 not in graph.adjacent:
                heurstic[node_2]=[random.uniform(40,47),random.uniform(20,29)]
                for i in range(0,max_connection_of_node):
                    selected_node_index_for_connection=list(graph.adjacent.keys())[random.randint(0,19)]
                    weight=random.randint(30,100)
                    if node_2 not in graph.adjacent:
                        graph.adjacent[node_2]=[]
                    edge1=edges(node_2,weight)
                    graph.adjacent[selected_node_index_for_connection].append(edge1)
                    edge2=edges(selected_node_index_for_connection,weight)
                    graph.adjacent[node_2].append(edge2)
                    array.append(selected_node_index_for_connection)
    def Add_random_nodes_and_draw_graph(self,graph):
        behch_marked_time_for_bfs=[]
        benchmarked_len_for_bfs=[]

        behch_marked_time_for_dfs=[]
        benchmarked_len_for_dfs=[]

        behch_marked_time_for_djkstra=[]
        benchmarked_len_for_djkstra=[]

        behch_marked_time_for_astar=[]
        benchmarked_len_for_astar=[]

        Benchmark1_for_bfs=self.Average_time_and_length_calculator_for_bfs(graph)
        Benchmark1_for_dfs=self.Average_time_and_length_calculator_for_dfs(graph)
        Benchmark1_for_djkstra=self.Average_time_and_length_calculator_for_djkstra(graph)
        Benchmark1_for_astar=self.performance_for_A_Star(graph)

        behch_marked_time_for_bfs.append(Benchmark1_for_bfs[0])
        benchmarked_len_for_bfs.append(Benchmark1_for_bfs[1])

        behch_marked_time_for_dfs.append(Benchmark1_for_dfs[0])
        benchmarked_len_for_dfs.append(Benchmark1_for_dfs[1])

        behch_marked_time_for_djkstra.append(Benchmark1_for_djkstra[0])
        benchmarked_len_for_djkstra.append(Benchmark1_for_djkstra[1])

        behch_marked_time_for_astar.append(Benchmark1_for_astar[0])
        benchmarked_len_for_astar.append(Benchmark1_for_astar[1])


        for i in range(0,19):
            self.generate_string_and_connect_randomly(graph)
        Benchmark2_for_bfs=self.Average_time_and_length_calculator_for_bfs(graph)
        Benchmark2_for_dfs=self.Average_time_and_length_calculator_for_dfs(graph)
        Benchmark2_for_djkstra=self.Average_time_and_length_calculator_for_djkstra(graph)
        Benchmark2_for_astar=self.performance_for_A_Star(graph)

        behch_marked_time_for_bfs.append(Benchmark2_for_bfs[0])
        benchmarked_len_for_bfs.append(Benchmark2_for_bfs[1])

        behch_marked_time_for_dfs.append(Benchmark2_for_dfs[0])
        benchmarked_len_for_dfs.append(Benchmark2_for_dfs[1])

        behch_marked_time_for_djkstra.append(Benchmark2_for_djkstra[0])
        benchmarked_len_for_djkstra.append(Benchmark2_for_djkstra[1])

        behch_marked_time_for_astar.append(Benchmark2_for_astar[0])
        benchmarked_len_for_astar.append(Benchmark2_for_astar[1])


        for i in range(0,19):
            self.generate_string_and_connect_randomly(graph)
        Benchmark3_for_bfs=self.Average_time_and_length_calculator_for_bfs(graph)
        Benchmark3_for_dfs=self.Average_time_and_length_calculator_for_dfs(graph)
        Benchmark3_for_djkstra=self.Average_time_and_length_calculator_for_djkstra(graph)
        Benchmark3_for_astar=self.performance_for_A_Star(graph)



        behch_marked_time_for_bfs.append(Benchmark3_for_bfs[0])
        benchmarked_len_for_bfs.append(Benchmark3_for_bfs[1])

        behch_marked_time_for_dfs.append(Benchmark3_for_dfs[0])
        benchmarked_len_for_dfs.append(Benchmark3_for_dfs[1])

        behch_marked_time_for_djkstra.append(Benchmark3_for_djkstra[0])
        benchmarked_len_for_djkstra.append(Benchmark3_for_djkstra[1])

        behch_marked_time_for_astar.append(Benchmark3_for_astar[0])
        benchmarked_len_for_astar.append(Benchmark3_for_astar[1])
        Number_of_nodes=[20,40,60]

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(19,5)) 

        ax[0].plot(Number_of_nodes,behch_marked_time_for_bfs,marker="o") 
        ax[0].plot(Number_of_nodes,behch_marked_time_for_dfs,marker="o") 
        ax[0].plot(Number_of_nodes,behch_marked_time_for_djkstra,marker="o") 
        ax[0].plot(Number_of_nodes,behch_marked_time_for_astar,marker="o")

 
        ax[0].legend(["Breadth First","Depth fIRST","Djkstra","A_star"])

        ax[0].set_xlabel('number of nodes') 
        ax[0].set_ylabel('Average time in Microseconds') 
        ax[0].set_title('Average time vs Number of nodes graph') 

        ax[1].plot(Number_of_nodes,benchmarked_len_for_bfs,marker="o") 
        ax[1].plot(Number_of_nodes,benchmarked_len_for_dfs,marker="o") 
        ax[1].plot(Number_of_nodes,benchmarked_len_for_djkstra,marker="o") 
        ax[1].plot(Number_of_nodes,benchmarked_len_for_astar,marker="o") 
        ax[1].legend(["Breadth First","Depth fIRST","Djkstra","A_star"])
        ax[1].set_xlabel('Number of nodes') 
        ax[1].set_ylabel('Average path length') 
        ax[1].set_title('Average path length vs number of nodes') 
        plt.show()


a=graph()



b=search()

b.Add_random_nodes_and_draw_graph(a)

print(b.BREAD_FIRST_SEARCH(a,"Oradea","Eforie"))
print(b.DEPTH_FIRST_SEARCH_ITERATIVE_SOLUTION(a,"Oradea","Eforie"))
print(b.DJKSTRA_SHORTEST_PATH(a,"Oradea","Eforie"))
print(b.A_star_algorithm(a,"Oradea","Eforie"))
