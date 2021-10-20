import pygame
import time


class fitness_sound():
    def __init__(self):
        self.soundon = 0
    
    def play_my_sound(self,type,timer):
        if timer < 10 and (self.soundon) == 0:
            file = f"sounds/{type}.wav"
            pygame.mixer.init()
            pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            self.soundon = 1
        else:
            pass

    # def sound_of_all():
    #     pygame.mixer.init()
        
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/bent_over_row.wav"))
    #     pygame.time.delay(26000)        
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/bicycle_crunch.wav"))
    #     pygame.time.delay(35000)
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/bridge.wav"))
    #     pygame.time.delay(23000)
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/dumbbell_bench_press.wav"))
    #     pygame.time.delay(45000)
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/hip_thrust.wav"))
    #     pygame.time.delay(57000)
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/lying_leg_raises.wav"))
    #     pygame.time.delay(23000)
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/side_lateral_raise.wav"))
    #     pygame.time.delay(38000)
    #     pygame.mixer.Sound.play(pygame.mixer.Sound("sounds/squat.wav"))
    #     pygame.time.delay(44000)

    #     return    

if __name__ == '__main__':
    fitness_sound.sound_of_all()





