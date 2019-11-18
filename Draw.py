import cv2
import math


def drawLines(image, lines, color=(0, 0, 0)):
    for x in range(0, len(lines) - 1):
        x0 = int(lines[x][0])
        y0 = int(lines[x][1])
        x1 = int(lines[x + 1][0])
        y1 = int(lines[x + 1][1])
        cv2.line(image, (x0, y0), (x1, y1), color, 3)
    return image


def drawSection(image, center, startangle, endAngle, collore, radius=50):
    center = (int(center[0]), int(center[1]))
    startangle = int(startangle)
    endAngle = int(endAngle)
    image = cv2.ellipse(image, center, (radius, radius), 0, startangle, endAngle, collore, 2)
    x1 = int(center[0] + radius * math.cos(math.radians(startangle)))
    y1 = int(center[1] + radius * math.sin(math.radians(startangle)))

    x2 = int(center[0] + radius * math.cos(math.radians(endAngle)))
    y2 = int(center[1] + radius * math.sin(math.radians(endAngle)))

    image = cv2.line(image, center, (x1, y1), collore, 2)
    image = cv2.line(image, center, (x2, y2), collore, 2)
    return image


if __name__ == "__main__":
    x = cv2.imread("..\\..\\Pictures/jallaball2.jpg")
    y = drawSection(x, (500, 500), -45, -135, (0, 255, 0))
    cv2.imshow("askdjhf", y)
    cv2.waitKey()
