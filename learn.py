# username =input('请输入用户名：')
# password =input('请输入口令：')

# if username == 'admin' and password =='123456':
#     print("successful")
# else:
#     print('fail')



#今天又是努力学习python的一天！！！
import turtle

# 设置画图的速度
turtle.speed(1)
# 抬起笔，移动坐标，设定初始坐标
turtle.up()
turtle.goto(-200, 200)
# 放下笔准备画
turtle.down()
# 准备填充颜色
turtle.begin_fill()
# 设置填充色和画笔的颜色
turtle.fillcolor('red')
turtle.pencolor('red')
# 使用循环，循环两次画出一个矩形
for i in range(2):
    turtle.forward(438)
    turtle.right(90)
    turtle.forward(292)
    turtle.right(90)
turtle.end_fill()

# 设置五角星颜色
turtle.fillcolor('yellow')
turtle.pencolor('yellow')

# 画大五角星
turtle.up()
turtle.goto(-170, 145)
turtle.down()
turtle.begin_fill()
for i in range(5):
    turtle.forward(50)
    turtle.right(144)
turtle.end_fill()

# 画第一颗小五角星
turtle.up()
turtle.goto(-100, 180)
# 调整画第一条线的角度
turtle.setheading(305)
turtle.down()
turtle.begin_fill()
for i in range(5):
    turtle.forward(20)
    turtle.right(144)
turtle.end_fill()

# 画第二颗小五角星
turtle.up()
turtle.goto(-85, 150)
turtle.setheading(30)
turtle.down()
turtle.begin_fill()
for i in range(5):
    turtle.forward(20)
    turtle.right(144)
turtle.end_fill()

# 画第三颗小五角星
turtle.up()
turtle.goto(-85, 120)
turtle.setheading(3)
turtle.down()
turtle.begin_fill()
for i in range(5):
    turtle.forward(20)
    turtle.right(144)
turtle.end_fill()

# 画第四颗小五角星
turtle.up()
turtle.goto(-100, 100)
turtle.setheading(300)
turtle.down()
turtle.begin_fill()
for i in range(5):
    turtle.forward(20)
    turtle.right(144)
turtle.end_fill()
# 隐藏箭头
turtle.hideturtle()
turtle.done()

turtle.mainloop()



