"""
THIS PROGRAM CONTAINS FUNCTIONS THAT CREATE FIGURES THAT GRAPHICALLY REPRESENT THE TEST RESULTS OF A LOCALIZATION STAGE
Among these graphics, we can find:
- Confusion matrices (coarse step)
- Maps with the network predictions (fine and global steps)
- Graphics that show the errors made by the network in every room (fine step)
Test programmes will call these functions
"""


import matplotlib.pyplot as plt
import os


def get_axes_limits(coordX, coordY, xmax, xmin, ymax, ymin):
    if coordX < xmin:
        xmin = coordX
    if coordX > xmax:
        xmax = coordX
    if coordY < ymin:
        ymin = coordY
    if coordY > ymax:
        ymax = coordY
    return xmax, xmin, ymax, ymin


def display_coord_map(direc, stage, mapVM, mapTest, k, ilum, r):

    xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
    plt.figure(figsize=(9, 5), dpi=120, edgecolor='black')

    firstk1, firstErrork, firstErrorRoom = True, True, True

    for vm in range(len(mapVM)):
        if vm == 0:
            plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue', label="Visual Model")
        else:
            plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue')
        xmax, xmin, ymax, ymin = get_axes_limits(mapVM[vm][0], mapVM[vm][1], xmax, xmin, ymax, ymin)

    for t in range(len(mapTest)):
        if mapTest[t][4] == "R":
            if firstErrorRoom:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='brown', label="Wrong Room")
                firstErrorRoom = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='brown')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], 'brown')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        elif mapTest[t][4] == "F":
            if firstErrork:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='red', label="Error")
                firstErrork = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='red')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], 'red')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        else:
            label = int(mapTest[t][4])
            if label < k/2:
                color = (2*label/k, 1, 0)
            elif label > k/2:
                color = (1, 1 - 2*(label - k/2) / k, 0)
            else:
                color = (1, 1, 0)
            if firstk1:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='green', label='Prediction amongst K=1')
                firstk1 = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], color=color)
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color=color)
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)

    plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.suptitle(stage + ' localization')
    plt.title(f'{r}, Illumination: {ilum}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(direc, "map_" + r + "_" + ilum + ".png"), dpi=400)
    plt.close()

    return
