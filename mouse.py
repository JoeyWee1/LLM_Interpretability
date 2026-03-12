import os
import time
import itertools

FRAMES = [
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)   (@)                               (o.o)
  />🍪     /|\                               /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)      (@)                            (o.o)
  />\         /|\                           /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)           (@)                       (o.o)
  />\              /|\                      /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)                (@)                  (o.o)
  />\                   /|\                 /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)                     (@)             (o.o)
  />\                        /|\            /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)                          (@)        (o.o)
  />\                             /|\       /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)                     (@)             (o.o)
  />\                        /|\            /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)                (@)                  (o.o)
  />\                   /|\                 /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)           (@)                       (o.o)
  />\              /|\                      /<\ 
""",
r"""
Two mice passing a cookie  (Ctrl+C to stop)

  (\_/)
  (o.o)      (@)                            (o.o)
  />\         /|\                           /<\ 
""",
]

def clear():
    os.system("cls" if os.name == "nt" else "clear")

try:
    for frame in itertools.cycle(FRAMES):
        clear()
        print(frame)
        time.sleep(0.15)
except KeyboardInterrupt:
    clear()
    print("bye")