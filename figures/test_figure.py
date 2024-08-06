import matplotlib.pyplot as plt

# Data
idx = list(range(50))
model_parameter_count = [448, 784, 4144, 1632, 0, 960, 0, 1552, 4144, 4144, 1632, 0, 0, 960, 0, 3104, 14432, 14432, 14432, 6336, 0, 1920, 160, 0, 9264, 30864, 30864, 14112, 0, 2880, 0, 23120, 82160, 0, 82160, 38880, 0, 4800, 0, 76960, 206080, 0, 0, 0, 136, 1281000, 432, 0, 480, 0]
number_of_rows_in_zk_circuit = [3644986, 1430026, 8888747, 3336721, 602112, 930147, 150528, 677380, 2211729, 2211728, 834182, 401408, 150528, 229920, 37632, 329281, 1697802, 1697804, 1697799, 718150, 75264, 336768, 1892423, 75264, 950215, 3449879, 3449878, 1528811, 112896, 168626, 28224, 583300, 2178096, 401408, 2178097, 1013328, 47040, 198480, 47040, 1916895, 5209716, 125440, 67200, 1280, 1204232, 643644, 4264972, 1204224, 1870903, 301056]

# Create figure and axis objects
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot model_parameter_count on primary y-axis
ax1.set_xlabel('Index')
ax1.set_ylabel('Model Parameter Count', color='tab:blue')
ax1.plot(idx, model_parameter_count, color='tab:blue', label='Model Parameter Count')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Set all idx values on x-axis
ax1.set_xticks(idx)

# Create secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Number of Rows in ZK Circuit', color='tab:orange')
ax2.plot(idx, number_of_rows_in_zk_circuit, color='tab:orange', label='Number of Rows in ZK Circuit')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Title and legend
plt.title('Model Parameter Count and Number of Rows in ZK Circuit by Index')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Show plot
plt.show()

pass

# "# Plot model_parameter_count on primary y-axis
# ax1.set_xlabel('Index')
# ax1.set_ylabel('Model Parameter Count', color='tab:blue')
# bar_width = 0.4
# bar1 = ax1.bar(np.array(idx) - bar_width/2, model_parameter_count, bar_width, color='tab:blue', label='Model Parameter Count')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Set all idx values on x-axis
# ax1.set_xticks(idx)

# # Create secondary y-axis
# ax2 = ax1.twinx()
# ax2.set_ylabel('Number of Rows in ZK Circuit', color='tab:orange')
# bar2 = ax2.bar(np.array(idx) + bar_width/2, number_of_rows_in_zk_circuit, bar_width, color='tab:orange', label='Number of Rows in ZK Circuit')
# ax2.tick_params(axis='y', labelcolor='tab:orange')

# # Title and legend
# plt.title('Model Parameter Count and Number of Rows in ZK Circuit by Index')
# fig.tight_layout()

# # Create legend
# fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# # Show plot
# plt.show()
# "