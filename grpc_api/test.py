import multiprocessing
import time

def heartbeat_process(_, status_file, stop_event, interval=2):
    print("[heartbeat] Process started", flush=True)
    while not stop_event.is_set():
        print("HEARTBEAT!", flush=True)
        stop_event.wait(interval)

if __name__ == "__main__":
    stop_event = multiprocessing.Event()
    p = multiprocessing.Process(target=heartbeat_process, args=(None, None, stop_event))
    p.start()
    time.sleep(5)
    stop_event.set()
    p.join()
