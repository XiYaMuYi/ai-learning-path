# -*- coding: utf-8 -*-
import os

if __name__ == '__main__':
    import sys

    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

    print(sys.path)

    r"""
    NOTE:
        python代码中进行import导入类的时候，实际上就是从sys.path中的路径进行查询，如果能够查询到，那么就可以正常运行，否则是无法运行的
        从sys.path中的第一个文件夹开始查，查到就结束
    1. PyCharm中默认会将project的根目录(当前就是: D:\workspaces\study\CV2502 )添加到sys.path环境变量中
    2. PyCharm会将__name__为__main__的py文件所在的文件夹添加到sys.path环境变量中(入口文件)
    3. PyCharm中可以通过右键选择文件夹，选择 Mark Directory As --> Sources Root 后，PyCharm会自动将该路径添加到sys.path中
    4. 通过python命令在命令行执行py文件的时候，python默认会将入口py文件所在的文件夹添加到sys.path中
    5. 实在不清楚的时候，通过sys.path.append进行环境路径的设定

    Python中的导包有两种方式：相对路径导包、绝对路径导包
        相对路径导包：相对于当前py文件来进行模块定位
            from .animal import Animal
            NOTE: 入口py文件中，不允许存在相对路径导包
        绝对路径导包：直接定位
            from zoo.animals.animal import Animal
    """

    from zoo.animals.cat import Cat
    from zoo.animals.dog import Dog

    # 下面导包如果要成功，必须将zoo.animals文件夹添加到sys.path中，通过方式3
    # from dog import Dog

    dog = Dog("发财")
    cat = Cat("吉祥")

    dog.eat()
    dog.say()
    cat.say()
    cat.eat()
