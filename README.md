# Droid Racing Challenge - Computer Vision Autonomous Navigation System

#### A computer vision system for autonomous circuit navigation via lane line perception.

## Introduction
The goal of this project is to create a computer vision algorithm that is suited to compete in QUT's DRC. As part of this competition, droids are raced in a circuit such as the one shown below.

*INSERT IMAGE HERE*

The rules of the competition also prohibits the use of any other sensors than a camera. As such, the navigation capabilities of the bot solely relies on the use of a computer vision algorithm. 

## Setup
For the purposes of the project, the team decided to use a Raspberry Pi that is connected to a web cam. The Raspberry Pi is then used to drive a steering servo that controls the the heading of the robot. The computer vision system shown here outputs a set of steering angles that are then translated to the correct movements by the Raspberry Pi. To run the computer vision system simply run:
```
python test_drive_v8.py
```
*Note: some packages that are used to control the servo and motor speeds may not be detected when used outside a Raspberry Pi. If this error occurs, simply delete these packages from test_drive_v8.py*  

## Concepts
The computer vision algorithm consists of four stages to calculate the correct steering angles our robot should take based on the lane lines it perceives. First, a filter is applied to extract the colors of the left and right lane lines (in this case blue and yellow). Then, an  edge detection algorithm is run to extract only the edges of the lane lines. The extracted edges are then fed to the Hough Transform to calculate the lines present in any given frame. Finally, the slope of the line is then determined and a heading line is taken from the averages of said lines. 

*Insert Image Here*
