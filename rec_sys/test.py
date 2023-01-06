import time

def print_info(i):
    print("第%d轮" % i)
    print('1234')


for i in range(1, 101):
    print("\r", end="")
    print_info(i)
    done = i//2
    undone = 50-done
    success_cnt = i
    fail_cnt = 0
    task_cnt = 100
    print("[{done}{undone}] success {s} | fail {f} | total {t}"
          .format(done=">" * done, undone=" " * undone, s=success_cnt, f=fail_cnt, t=task_cnt),
          end='')
    time.sleep(1)

