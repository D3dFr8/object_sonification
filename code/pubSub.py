from threading import Thread
import point
import getHRIR
import soundTools
import asyncio

class SoundChannel:
    y = []

    def empty(self):
        return len(self.y) == 0

def child(conn): #target, audio, sr, yL, yR, ready, running
    target, audio, sr, yL, yR, ready, running = conn.recv()
    while running[0]:
        if ready[0]:
            hL,hR = getHRIR.getHrirAtTarget(target[0])
            yL.y = soundTools.conv(audio,hL).tolist()
            yR.y = soundTools.conv(audio,hR).tolist()
            asyncio.run(soundTools.playSound(yL.y,yR.y,sr))
            target.pop()
            ready[0] = False
            conn.send([target, audio, sr, yL, yR, ready, running])
