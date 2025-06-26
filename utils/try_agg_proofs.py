import ezkl
import os

# Use a raw string (r"...") or double backslashes for Windows paths
directory = r"utils/proofs_to_aggregate"

proofs = [f for f in os.listdir(directory) if f.endswith('.json')]




# # now mock aggregate the proofs
# proofs = []
# for i in range(3):
#     proof_path = os.path.join('proof_split_'+str(i)+'.json')
#     proofs.append(proof_path)

result = ezkl.aggregate(proofs)
print("Aggregated proof:", result)