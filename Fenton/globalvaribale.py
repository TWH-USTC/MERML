def init():
    global global_variable_dict
    global_variable_dict = {}


def set_value(key, value):
    # 定义一个全局变量
    global_variable_dict[key] = value


def get_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return global_variable_dict[key]
    except KeyError:
        print("This key is not in the global variable dictionary")


def get_dict():
    return global_variable_dict
