import json
import matplotlib.pyplot as plt

# Sample JSON data as a string (replace with actual file reading)
data = """
{"timestamp": 1741708450.65407, "elapsed_time": 1.0735604763031006, "CPU Usage (%)": 14.4}
{"timestamp": 1741708452.6954372, "elapsed_time": 3.1149277687072754, "CPU Usage (%)": 17.3}
{"timestamp": 1741708454.7363443, "elapsed_time": 5.155834436416626, "CPU Usage (%)": 20.1}
{"timestamp": 1741708456.775709, "elapsed_time": 7.1951987743377686, "CPU Usage (%)": 20.0}
{"timestamp": 1741708458.9321973, "elapsed_time": 9.351687669754028, "CPU Usage (%)": 6.3}
{"timestamp": 1741708460.9711378, "elapsed_time": 11.390628099441528, "CPU Usage (%)": 3.2}
{"timestamp": 1741708463.0754268, "elapsed_time": 13.494916915893555, "CPU Usage (%)": 5.4}
{"timestamp": 1741708465.1172605, "elapsed_time": 15.53675103187561, "CPU Usage (%)": 30.5}
{"timestamp": 1741708467.1896496, "elapsed_time": 17.609139442443848, "CPU Usage (%)": 49.2}
"""

# Load JSON lines
lines = data.strip().split("\n")
records = [json.loads(line) for line in lines]

# Extract elapsed_time and CPU Usage
elapsed_time = [rec["elapsed_time"] for rec in records]
cpu_usage = [rec["CPU Usage (%)"] for rec in records]

# Plot CPU Usage over time
plt.figure(figsize=(8, 5))
plt.plot(elapsed_time, cpu_usage, marker='o', linestyle='-', color='b', label="CPU Usage (%)")

plt.xlabel("Elapsed Time (s)")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage Over Time")
plt.grid(True)
plt.legend()
plt.show()
