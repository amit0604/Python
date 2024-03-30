graph = {
  '3' : ['5','1'],
  '5' : ['2', '4'],
  '1' : ['6'],
  '2' : [],
  '4' : ['6'],
  '6' : []
}

def dfs_traversal(visited, graph, node): 
    visited.append(node)
    print(node, end=" ")

    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs_traversal(visited, graph, neighbour)

def print_leaf_nodes_dfs(visited, graph, node): 
    visited.append(node)

    # Check if the node is a leaf node
    if not graph[node]:
        print(node, end=" ")
    else:
        for neighbour in graph[node]:
            if neighbour not in visited:
                print_leaf_nodes_dfs(visited, graph, neighbour)

# Driver Code
print("DFS Traversal order:")
dfs_traversal([], graph, '3')
print("\nLeaf nodes in the graph are:")
print_leaf_nodes_dfs([], graph, '3')