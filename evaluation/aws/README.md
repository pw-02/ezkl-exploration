# AWS scaling from your PC

Use this independent AWS runner while the EPYC experiments continue. It uses
your AMI `ami-07f7ea3891a7d829e`, region `us-east-1`, security group
`sg-09701eae1bec5180d`, `/home/ubuntu/zkinfer`, Conda environment `zk`, and
bucket `zkinfer`. It does not run or alter your local proof campaign.

The PC needs Python, Git and AWS credentials; it does **not** need EZKL or models.
The AMI needs your working backend environment and model/input files. Source and
configuration files are uploaded from your PC checkout to the new experiment
instances, so the AMI need not contain these new scripts or GitHub credentials.
This overlays tracked `zkinfer/`, `evaluation/`, and `experiments/config/` files;
it does not reinstall your custom EZKL wheel. Use dedicated experiment instances.

## What SSM and the instance role mean

Your administrator login authorizes commands **from your PC**. It is not inherited
by an EC2 machine. An **instance role** gives that machine its own temporary AWS
credentials. The `init` command creates a dedicated `zkinfer-eval-worker` role and
instance profile with:

- `AmazonSSMManagedInstanceCore`, to receive Systems Manager commands;
- read/write/delete access only under `s3://zkinfer/evaluation/`, plus bucket
  location and prefix-limited listing. It does not grant administrator access.

**SSM Agent** is the service on Ubuntu that receives those commands. Launch user
data starts an existing package/snap or installs the Ubuntu snap if absent. No
Session Manager plugin is needed on your PC: this uses Run Command, not a shell
session. Do not bake access keys into the image.

Instances need outbound HTTPS to SSM and S3 (and the snap store if installing the
agent). The default configuration requests public IPs in one default subnet;
that subnet still needs an internet-gateway route. Private subnets instead need
appropriate NAT or VPC endpoints. The script does not create routing/NAT/endpoints.
`init` adds TCP 50051 ingress **from the same supplied security group only**;
it never opens the coordinator to the internet. Existing group rules are preserved.
Because you supplied the default group, this rule applies to other members too;
you may substitute a dedicated group in the same VPC before initialization.

AWS references:
[SSM Agent](https://docs.aws.amazon.com/systems-manager/latest/userguide/ssm-agent-linux.html),
[instance permissions](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-permissions.html),
[network connectivity](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-create-vpc.html).

## 1. On your PC: get the code and configure AWS

In **Windows PowerShell**, from the repository checkout:

```powershell
git switch sacdad27
git pull --ff-only
py -m pip install boto3
aws sts get-caller-identity --region us-east-1
```

Use your normal AWS CLI login/profile. For IAM Identity Center, log in with
`aws sso login --profile YOUR_PROFILE` and add `--profile YOUR_PROFILE` to each
controller command below. Never send credentials in chat or commit them.
In WSL/Linux replace `py` with `python3`; the same controller works on both.

Inspect `evaluation/aws/config.json` **before launching**:

| Setting | Initial value / meaning |
|---|---|
| `nodes` | `[1, 2, 4, 6, 8]` physical prover nodes, including coordinator's worker |
| `workloads` | `mnist_gan` first; add other config stems before starting a new campaign |
| `requests` | `[1, 16]`, fixed total requests at every node count |
| `trials` | 1 initial pass, no variability/error-bar claim |
| `instance_type` | `m6a.4xlarge` |
| `root_volume_gib` | **1000 GiB gp3 per instance** for artifacts/cache; adjust deliberately |
| `memory_budget_gib` | 48 GiB logical admission budget per node; require at least 8 GiB actual headroom |
| `threads` | 8 Rayon threads per prover, fixed across profiles and trials; not a tuned optimum |
| `cache_gib` | 500 GiB disk cache, not RAM |
| `timeout_seconds` | 21600 (six hours per profiling action / measured batch) |
| `subnet_id` | Blank selects one default subnet in the security group's VPC; set explicitly if needed |
| `conda_executable` | Blank tries miniconda3, anaconda3, miniforge3 under `/home/ubuntu`, then `/opt/conda` |
| `campaign` | `aws-scale-v1`, isolates S3 objects and instance tags |

Storage is deliberately explicit: 1000 GiB on eight nodes means eight volumes.
Stopping instances retains chargeable EBS volumes and all S3 data. Nothing here
terminates instances or deletes volumes. Check your account's instance/vCPU quota
before requesting eight nodes (128 vCPUs for this instance type).

## 2. Preview, create the role, and launch just one node

```powershell
py evaluation/aws/control.py run
py evaluation/aws/control.py init --execute
py evaluation/aws/control.py launch --nodes 1 --execute
```

Without `--execute`, commands print the plan and make **no AWS calls**. `init`
creates/attaches the dedicated role policies and the internal security-group rule.
If IAM propagation delays the first launch, wait briefly and repeat the command.
`launch` waits for EC2 health checks **and SSM Online**, rather than treating
instance creation as readiness. The AMI must be available first.

Only instances tagged with this exact `zkInferCampaign` are managed. Existing
manually launched instances and your reference/builder instance are not adopted.
The launch command resumes stopped campaign instances or adds missing indices;
it does not recreate existing workers. Do not manually terminate campaign nodes
and then reuse the same campaign name/client tokens.

## 3. Run the campaign

```powershell
py evaluation/aws/control.py run --execute
```

The runner:

1. Uploads a source bundle, freezes its hash and the configuration, and checks
   the selected model/input hashes, backend version, RAM and CPUs on each worker.
2. On node 1, generates setup once without proofs, then collects **one full warm
   profiling request per workload** using the existing isolated campaign helper.
   The profiles are collected on AWS, not copied from the EPYC host. These initial
   resource profiles use local setup files; S3 restoration is part of measured
   trials. This is not a repeatability microbenchmark or a storage performance model.
3. Uploads reusable setup, SRS, and both profile snapshots to a campaign seed in S3.
4. Runs whole-job and stage admission at each node count; adds workers only as the
   ascending sweep needs them. Node 1 hosts the coordinator and a prover worker.
5. Gives each attempt fresh local caches/state and fresh writable coordinator
   profiles in S3. All workers receive identical initial node histories. Setup is
   available remotely in S3, **not preloaded in worker caches**. SRS is copied before
   measurement. OS page caches are not flushed.
6. Waits until the coordinator has observed exactly the expected live worker IDs,
   then submits traffic on node 1. Provisioning/profile collection is outside timing.
   All requests must complete, with runtime verification enabled, for a success marker.
7. Stops runtime services, uploads reports/logs, then marks the cell complete.
   After the full campaign succeeds, it stops all campaign instances.

The default matrix is **20 measured cells**: five node counts × two batch sizes ×
two policies. Extend to all three primary models only when time permits:
`mnist_gan`, `mobilenet_v2`, `nano_gpt_4_layers_64_embd`. Keep 1 node for the matched
AWS speedup denominator. Ten nodes is optional, not required.

**Keep your PC awake and connected for this controller.** Active remote commands
and systemd services do not depend on an open SSH session, but your PC orchestrates
subsequent steps and result collection. Closing it does not make the rest of the
sweep continue automatically. Services have a finite runtime watchdog; workers
are not automatically powered off if the PC disconnects. Unattended progression
would require placing the controller on a persistent management host with its own
EC2/SSM orchestration permissions; that is not granted to prover workers here.

## 4. Results, restart, and stopping

```powershell
py evaluation/aws/control.py status --execute
aws s3 ls s3://zkinfer/evaluation/aws-scale-v1/ --recursive
aws s3 sync s3://zkinfer/evaluation/aws-scale-v1/completed/ evaluation/results/aws-completed/
```

S3 layout:

- `source/` and `campaign.json`: source hash and frozen settings;
- `checks/`: per-instance environment/model checks;
- `seeds/<model>/`: setup, SRS, immutable initial snapshots, profiling reports;
- `runs/<cell>/<attempt>/`: S3 request artifacts, mutable coordinator profiles,
  and `nodes/<instance-id>/` with raw reports, traffic summary and logs;
- `completed/<cell>.json`: successful, collected attempt reference;
- `ssm/`: remote command stdout/stderr.

Run the same command again to skip completed cells. Configuration/source changes
require a new campaign name; do not combine results from different settings.
The PC maintains checkpoints under `evaluation/results/aws-control/<campaign>/`.
Keep that directory and use one controller at a time, on the same PC checkout.

After an interruption or failure during a measured trial:

```powershell
py evaluation/aws/control.py recover --execute
py evaluation/aws/control.py run --execute
```

Recovery cancels outstanding traffic, stops only that attempt's runtime services,
and collects available reports. It retains the interrupted attempt without marking
it successful. The next run retries with a new attempt prefix. A failed setup/profile
action can be rerun after its SSM command is terminal; do not start a second while
the first is still running. The console prints each SSM command ID, which you can
inspect in Systems Manager Run Command.

If collection fails, do not stop instances until you have recovered their local
logs. For an explicit stop (not termination):

```powershell
py evaluation/aws/control.py stop --execute
```

An incomplete campaign does not automatically stop instances, to preserve access
for diagnosis. Check the EC2 console after stopping. Remove unused EBS snapshots,
volumes and AMIs separately only when you no longer need them.

## Troubleshooting before spending time on proofs

- **SSM not Online:** inspect `/var/log/cloud-init-output.log`; check
  `sudo snap services amazon-ssm-agent` or `sudo systemctl status amazon-ssm-agent`.
  Check the attached IAM instance profile and outbound HTTPS routing.
- **Conda not found:** on the working reference node run `echo "$CONDA_EXE"`, then
  put that absolute executable path in `conda_executable` before the campaign.
- **boto3 missing in `zk`:** install it once in that environment on the reference
  node and create an updated AMI. The runner does not replace backend packages.
- **S3 access denied:** inspect bucket policy and any KMS key policy. This role
  grants access to `evaluation/` only; it does not override bucket explicit denies
  or grant access to a customer-managed KMS key.
- **Port already in use:** stop any old runtime auto-start service baked into the
  AMI. The environment check refuses to reuse a busy coordinator port.
- **Models missing:** workload YAML paths must resolve on the AMI. Models and
  inputs are not uploaded from your PC by this source-only deployment.
- **Disk fills:** the reference node retains setup and profiling output. Increase
  disk capacity or reduce the workload list; cache size alone is not a disk budget.

## Validation scope

```powershell
py -m unittest discover -s evaluation/aws -p "test_*.py" -v
```

Tests exercise configuration, isolation and failure handling with mocks. No EC2
instances or real proofs were launched when these scripts were authored. Start
with the one-node cell and inspect its reports before relying on the scaling data.
