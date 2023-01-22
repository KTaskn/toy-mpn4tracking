import numpy as np
from uuid import uuid4

print("id,bf,time,red,green,blue,x,y")

def randomwalk(rgb, xy):
    rgb = np.random.multinomial(255, np.array(rgb) / np.sum(rgb))    
    x = xy[0] + np.random.normal(1, 1) // 1
    y = xy[1] + np.random.normal(1, 1) // 1
    
    return rgb, [x, y]


def gendata(N):
    rgb, xy = np.random.multinomial(5, [1/3, 1/3, 1/3]), np.random.randint(0, 100, 2)

    lrgb, lxy = [], []
    for _ in range(N):
        rgb, xy = randomwalk(rgb, xy)
        lrgb.append(rgb)
        lxy.append(xy)

    lt = range(N)
    lrgb = np.array(lrgb) / 255
    lxy = np.array(lxy)
    lid = ["#"] + [str(uuid4())for _ in range(N)]

    for t, (r, g, b), (x, y), id, bf in zip(lt, lrgb, lxy, lid[1:], lid[:-1]):
        print(f"{id},{bf},{t},", end="")
        x, y = int(x), int(y)
        print(f"{r:.3},{g:.3},{b:.3},{x},{y}")

if __name__ == "__main__":
    N, G = 20, 3
    for i in range(G):
        gendata(N)