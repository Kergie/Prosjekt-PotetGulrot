"""
Path planning Sample Code with RRT*
author: Atsushi Sakai(@Atsushi_twi)
"""

import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from maze_recogn import mazeRecognizer
from Dead import Dead
from Draw import drawLines
import cv2


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = False
show_final_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, lineList, edge_dist, rand_area_x, rand_area_y,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=20,
                 max_iter=10000,
                 connect_circle_dist=50.0
                 ):
        """

        :param start: Coordinates for start point
        :param goal: Coordinates for goal point
        :param lineList: list of lines for obstacles (x1y1,x2y2 coordinates)
        :param edge_dist: distance to keep from the obstacles
        :param rand_area_x: area in the x-plane for which the nodes can be placed
        :param rand_area_y: area in the y-plane for which the nodes can be placed
        :param expand_dis: expand distance
        :param path_resolution: path resolution, stepsize
        :param goal_sample_rate: chance for it to try to just go to goal
        :param max_iter: max iterations
        :param connect_circle_dist: size around for which it will search for nodes
        """
        super().__init__(start, goal,
                         rand_area_x, rand_area_y, lineList, edge_dist, expand_dis, path_resolution, goal_sample_rate,
                         max_iter)

        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.showFinalAnimation = True

    def planning(self, animation=False, search_until_max_iter=False):
        """
        rrt star path planning
        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)

            if self.checkObstaclev2(new_node, self.lineList, self.edge_dist):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    print("Iterations: ", i)
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.checkObstaclev2(t_node, self.lineList, self.edge_dist):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.checkObstaclev2(t_node, self.lineList, self.edge_dist):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        Searches for nearby nodes

        :param new_node: node for which to search around
        :return: the index of the closest nodes
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # print(r)
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
        Tries to rewire the path to be more cost efficient

        :param new_node: node for which to try to rewire
        :param near_inds: indexes of nearby nodes
        :return: nothing
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.checkObstaclev2(edge_node, self.lineList, self.edge_dist)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        """
        Calculates new cost of the path (the distance)

        :param from_node: from node
        :param to_node: to node
        :return: new node cost for the from-node
        """
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def run(self, finishLoops=False):
        """
        Runs the RRT_Star-class.

        :return: picture of the figure as a NP-array with the path
        """
        path = self.planning(animation=False, search_until_max_iter=finishLoops)

        fig = None

        if path is None:
            print("Cannot find path")
            fig = plt.figure()
            fig.add_subplot(111)
            self.draw_graph()
        else:
            print("found path!!")

            # Draw final path
            if self.showFinalAnimation:
                self.draw_graph()
                fig = plt.figure()
                fig.add_subplot(111)
                plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                for (data) in self.lineList:
                    x1 = data[0][0]
                    y1 = data[0][1]
                    x2 = data[0][2]
                    y2 = data[0][3]
                    self.plotObstaclev2(x1, y1, x2, y2)
                plt.grid(True)
                plt.pause(0.01)  # Need for Mac
                plt.show()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data, path


class multiRRTStar:

    def __init__(self, rand_area_x=None, rand_area_y=None, lineList=None, expand_dis=100.0,
                 path_resolution=10.0, max_iter=2000, goal_sample_rate=30, edge_dist=30, connect_circle_dist=450,
                 start_point=None, listOfDeadEnds=None):

        if rand_area_y is None:
            rand_area_y = [0, 1100]
        self.rand_area_x = rand_area_x
        self.rand_area_y = rand_area_y
        self.lineList = lineList
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.edge_dist = edge_dist
        self.connect_circle_dist = connect_circle_dist
        self.start_point = start_point
        self.listOfDeadEnds = listOfDeadEnds

    def tempName(self, startpoint, pointList: list):
        paths = []

        for points in pointList:
            rrtStar = RRTStar(start=[startpoint[0], startpoint[1]], goal=[points[0], points[1]],
                              rand_area_x=self.rand_area_x, rand_area_y=self.rand_area_y,
                              lineList=self.lineList, expand_dis=self.expand_dis, path_resolution=self.path_resolution,
                              max_iter=self.max_iter, goal_sample_rate=self.goal_sample_rate,
                              edge_dist=self.edge_dist, connect_circle_dist=self.connect_circle_dist)
            _, path = rrtStar.run(finishLoops=False)
            paths.append(path[::-1])

        return paths

    def sumPaths(self, pathList):
        listOfLengths = []

        for paths in pathList:
            sum = 0
            if paths is None:
                listOfLengths.append(float("inf"))
            else:
                for i in range(len(paths) - 1):
                    sum += math.sqrt((paths[i+1][0] - paths[i][0]) ** 2 + (paths[i+1][1] - paths[i][1]) ** 2)
                listOfLengths.append(sum)

        return listOfLengths

    def run(self):
        finalPath = []

        deadEndList = self.listOfDeadEnds.copy()

        newStartPoint = self.start_point

        while len(deadEndList) > 0:
            pathList = self.tempName(newStartPoint, deadEndList)

            pathSumList = self.sumPaths(pathList)

            indexNewStartPoint = pathSumList.index(min(pathSumList))

            finalPath.append(pathList[indexNewStartPoint])

            newStartPoint = deadEndList.pop(indexNewStartPoint)

        return finalPath






if __name__ == "__main__":
    enkelRRT = False
    multidriftRRT = True

    if enkelRRT:
        m = mazeRecognizer()
        lines, _ = m.findMaze()
        startPoints = [[830, 365], [720, 450], [840, 870], [1250, 250]]
        endPoints = [[720, 450], [840, 870], [1250, 250], [1400, 150]]

        finalPath = []
        # rrt_star = RRTStar(start=[startPoints[3][0], startPoints[3][1]], goal=[endPoints[3][0], endPoints[3][1]], rand_area_x=[500, 1600], rand_area_y=[0, 1100],
        #                   lineList=lines, expand_dis=100.0, path_resolution=10.0, max_iter=2000, goal_sample_rate=30,
        #                   edge_dist=30, connect_circle_dist=450)

        rrt_star = None

        for (data1, data2) in zip(startPoints, endPoints):
            startx = data1[0]
            starty = data1[1]
            goalx = data2[0]
            goaly = data2[1]
            rrt_star = RRTStar(start=[startx, starty], goal=[goalx, goaly], rand_area_x=[500, 1600], rand_area_y=[0, 1100],
                               lineList=lines, expand_dis=100.0, path_resolution=10.0, max_iter=2000, goal_sample_rate=30,
                               edge_dist=30, connect_circle_dist=450)
            path = rrt_star.planning()
            print()
            path = path[::-1]
            for pathPoints in path:
                finalPath.append(pathPoints)

        # path = rrt_star.planning()

        showFinalAnimation = True

        if finalPath is None:
            fig = plt.figure()
            fig.add_subplot(111)
            rrt_star.draw_graph()
        else:
            print("found path!!")

            # Draw final path
            if showFinalAnimation:
                rrt_star.draw_graph()
                fig = plt.figure()
                fig.add_subplot(111)
                plt.plot([x for (x, y) in finalPath], [y for (x, y) in finalPath], '-r')
                for (data) in rrt_star.lineList:
                    x1 = data[0][0]
                    y1 = data[0][1]
                    x2 = data[0][2]
                    y2 = data[0][3]
                    rrt_star.plotObstaclev2(x1, y1, x2, y2)
                plt.grid(True)
                plt.pause(0.01)  # Need for Mac
                plt.show()
    elif multidriftRRT:
        m = mazeRecognizer()
        lines, _ = m.findMaze()
        d = Dead()
        bilde = cv2.imread(os.getcwd() + "\\" + "..\\..\\Pictures\\DeadEnds\\perf2.jpg")
        listOfDeadEnds, _ = d.getDeadEnds2(bilde)

        startPoint = [1280, 300]

        r = multiRRTStar(rand_area_x=[500, 1600], rand_area_y=[0, 1100],
                               lineList=lines, expand_dis=100.0, path_resolution=10.0, max_iter=2000, goal_sample_rate=30,
                               edge_dist=30, connect_circle_dist=800, start_point=startPoint, listOfDeadEnds=listOfDeadEnds)

        fPath = r.run()

        finalFinalPath = []

        for paths in fPath:
            for data in paths:
                finalFinalPath.append(data)

        sisteBilde = drawLines(bilde, finalFinalPath, (255, 0, 0))

        cv2.imwrite("finalpicture.jpg", sisteBilde)

        cv2.imshow("Yolo", sisteBilde)
        cv2.waitKey()