graph = {
  '3' : ['5','1'],
  '5' : ['2', '4'],
  '1' : ['6'],
  '2' : [],
  '4' : ['6'],
  '6' : []
}

visited = [] # List for visited nodes.
queue = []   # Initialize a queue

def bfs_traversal(visited, graph, node): # function for BFS traversal
  visited.append(node)
  queue.append(node)

  while queue:          # Creating loop to visit each node
    m = queue.pop(0)
    print(m, end=" ")
    
    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

def print_leaf_nodes(visited, graph, node): # function to print leaf nodes
  visited.append(node)
  queue.append(node)

  while queue:
    m = queue.pop(0)
    
    # Check if the node is a leaf node
    if not graph[m]:
        print(m, end=" ")
    else:
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Driver Code
print("BFS Traversal order:")
bfs_traversal([], graph, '3')
print("\nLeaf nodes in the graph are:")
print_leaf_nodes([], graph, '3')