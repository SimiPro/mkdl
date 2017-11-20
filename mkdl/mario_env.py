import subprocess
import multiprocessing
import time
import mss
import mss.tools
import os


M_WIDTH = 320
M_HEIGHT = 240


def run_mario():
    subprocess.run(['mupen64plus', '--windowed', '--resolution', "{}x{}".format(M_WIDTH, M_HEIGHT), 'm_kart_64.z64'])


if __name__ == '__main__':
    mario_proc = multiprocessing.Process(target=run_mario, name="run mario")

    mario_proc.start()

    print("after run mario try capturing the screen: put mario to the top left corner pls")


    for i in range(10):
        time.sleep(2) # lets wait till mario is set up

        with mss.mss() as sct:
            # The screen part to capture
            monitor = {'top': 0, 'left': 0, 'width': M_WIDTH, 'height': M_HEIGHT, 'i':i}
            output = 'sct-{top}x{left}_{width}x{height}_{i}.png'.format(**monitor)

            # Grab the data
            sct_img = sct.grab(monitor)

            # Save to the picture file
            mss.tools.to_png(sct_img.rgb, sct_img.size, output)
            print(output)
