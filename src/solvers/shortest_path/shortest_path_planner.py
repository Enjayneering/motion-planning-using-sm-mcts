
class ShortestPathPlanner:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal

    def plan(self):
        return nx.shortest_path(self.graph, self.start, self.goal)