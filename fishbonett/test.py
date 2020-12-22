from fishbonett.fishbone.model import _c
import numpy as np

class C:
    def __init__(self):
        self._x = [1, 2, 3]

    @property
    def x(self):
        print("getter")
        return self._x
    # @x.setter
    # def x(self, value):
    #     print("setter")
    #     self._x = value


print(_c(2))

def eye(d):
    if d is [] or None:
        print("1",d)
        return None
    elif type(d) is int or d is str:
        print("2",d)
        return np.eye(int(d))
    elif type(d) is list:
        print("3", d)
        return np.eye(*d)

print("eye3=", eye(3))
# a = [3, 3, 3]
# b = [2]
# pd = np.array([[a, b, b, a], [a, b, b, []]], dtype=object)
# # TODO handle the case above
# tri = FishBoneH(pd)
# tri.domain = [-1, 1]
# print("sd", tri.sd)
# print("sd",tri.sd)
#
# print("evL",tri._evL)
# print("evL",tri._ebL)
# print("!",tri._evL)
# tri.build_coupling()
# print(tri.k_list)
# print(tri.w_list)