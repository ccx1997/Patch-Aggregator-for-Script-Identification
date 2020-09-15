# -*- coding: utf-8 -*-
# By Changxu Cheng, HUST

from __future__ import division
import  numpy as np
from skimage import io, color
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def drawCM(matrix, savname):
    # Display different color for different elements
    lines, cols = matrix.shape
    sumline = matrix.sum(axis=1).reshape(lines, 1)
    ratiomat = matrix / sumline
    toplot0 = 1 - ratiomat
    toplot = toplot0.repeat(50).reshape(lines, -1).repeat(50, axis=0)
    io.imsave(savname, color.gray2rgb(toplot))
    # Draw values on every block
    image = Image.open(savname)
    draw = ImageDraw.Draw(image)
    if __name__ == "__main__":
        fontdir = "ARIAL.TTF"
    else:
        fontdir = os.path.join(os.getcwd(), "draw/ARIAL.TTF")
    font = ImageFont.truetype(fontdir, 15)
    for i in range(lines):
        for j in range(cols):
            dig = str(matrix[i, j])
            if i == j:
                filled = (255, 181, 197)
            else:
                filled = (46, 139, 87)
            draw.text((50 * j + 10, 50 * i + 10), dig, font=font, fill=filled)
    image.save(savname, 'jpeg')

def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), fontdict={'fontsize': 8}, va='center', ha='center')
    ax.set_xticklabels([''] + classes, fontdict={'fontsize': 10}, rotation=90)
    ax.set_yticklabels([''] + classes)
    #save
    plt.savefig(savname)


if __name__ == "__main__":
    matrix = np.random.randint(16, size=16).reshape(4,4)
    savname1 = 'tmp1.jpg'
    savname2 = 'tmp2.jpg'
    drawCM(matrix, savname1) # draw without axis
    classes = [chr(ord('a')+i) for i in range(4)]
    plotCM(classes, matrix, savname2)

