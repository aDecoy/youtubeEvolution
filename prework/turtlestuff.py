import turtle
wn = turtle.Screen()
turtle.speed(0)

# turtle.color('red', 'yellow')
# turtle.begin_fill()
a = 10
# for _ in range(100)  :
#     turtle.forward(200*a)
#     turtle.left(110)
#     turtle.back(10*a)
#     turtle.left(11)
# for _ in range(50)  :
while True:
    turtle.forward(20*a)
    turtle.left(91)
    turtle.forward(90*a)
    turtle.left(91)

    # turtle.back(20 * a)

    # turtle.back(10*a)
    # for _ in range(7):
    #     turtle.forward(10 * a)
    #     turtle.left(120)

# turtle.end_fill()

from sys import platform
if platform=='win32':
    wn.exitonclick()
# done()