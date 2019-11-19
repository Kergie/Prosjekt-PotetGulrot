from Dead import Dead
from Draw import drawLines
from findSnake import FindSnake
from findTarget import FindTarget
from maze_recogn import mazeRecognizer
from rrt_star import multiRRTStar, RRTStar
import cv2
import os

multi = True
single = False

mazePic = cv2.imread(os.getcwd() + "\\Pictures\\onlyMazeTest2.jpg", -1)
snakeTargetPic = cv2.imread(os.getcwd() + "\\Pictures\\snakeAndTarget.png", -1)
onlySnakePic = cv2.imread(os.getcwd() + "\\Pictures\\onlySnake.png", -1)

if multi:
    mr = mazeRecognizer()
    deadends = Dead()
    fs = FindSnake()

    snakeCoordinates, _ = fs.LocateSnakeAverage(1, 1, picture=onlySnakePic)

    lines, _ = mr.findMaze(mazePic)
    listOfDeadEnds, _ = deadends.getDeadEnds2(mazePic)

    startPoint = [snakeCoordinates[1][0], snakeCoordinates[1][1]]

    r = multiRRTStar(rand_area_x=[500, 1600], rand_area_y=[0, 1100], lineList=lines, expand_dis=100.0,
                     path_resolution=10.0, max_iter=2000, goal_sample_rate=30, edge_dist=30, connect_circle_dist=800,
                     start_point=startPoint, listOfDeadEnds=listOfDeadEnds)

    fPath = r.run()
    finalFinalPath = []

    for paths in fPath:
        for data in paths:
            finalFinalPath.append(data)

    sisteBilde = drawLines(snakeTargetPic, finalFinalPath, (255, 0, 0))

    cv2.imshow("Yolo", sisteBilde)
    cv2.waitKey()

elif single:
    mr = mazeRecognizer()
    deadends = Dead()
    fs = FindSnake()
    ft = FindTarget()

    d, frame, radius, center = ft.getTarget(snakeTargetPic)

    lines, _ = mr.findMaze(mazePic)

    snakeCoordinates, _ = fs.LocateSnakeAverage(1, 1, picture=onlySnakePic)
    startPoint = [snakeCoordinates[1][0], snakeCoordinates[1][1]]

    r = RRTStar(start=startPoint, goal=[center[0], center[1]], rand_area_x=[250, 1500],
                rand_area_y=[0, 1100],
                lineList=lines,
                expand_dis=100.0, path_resolution=10.0, max_iter=2000, goal_sample_rate=20,
                connect_circle_dist=800,
                edge_dist=30)

    _, path = r.run(finishLoops=False)

    sisteBilde = drawLines(snakeTargetPic, path, (255, 0, 0))

    cv2.imshow("Yolo", sisteBilde)
    cv2.waitKey()
