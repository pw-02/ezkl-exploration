#!/usr/bin/env python3
"""PC-side AWS campaign controller: Python + boto3, works in PowerShell or WSL.

No cloud writes unless --execute is supplied. No instances are terminated.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def validate(c):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', c['campaign']): raise ValueError('Invalid campaign name')
    if not c['repo'].startswith('/home/ubuntu/'): raise ValueError('Expected ubuntu-owned repo path')
    if c['nodes'] != sorted(set(c['nodes'])) or any(n<1 or n>10 for n in c['nodes']):
        raise ValueError('Node counts must be increasing unique integers in 1..10')
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]+', m) for m in c['workloads']): raise ValueError('Invalid workload')
    if not c['workloads'] or not c['requests'] or not c['nodes']: raise ValueError('Empty matrix')
    if min(c['requests'])<1 or c['trials']<1 or c['threads']<1: raise ValueError('Invalid counts')
    if not 60 <= c['timeout_seconds'] <= 169000: raise ValueError('Timeout outside supported SSM range')
    if c['memory_budget_gib']<=0 or c['cache_gib']<=0: raise ValueError('Invalid memory/cache budget')


def cells(c, nodes):
    for model in c['workloads']:
        for trial in range(1, c['trials']+1):
            policies = ['whole-job','stage'] if trial%2 else ['stage','whole-job']
            for requests in c['requests']:
                for baseline in policies:
                    key = f'{model}-{baseline}-nodes{nodes}-requests{requests}-r{trial}'
                    yield key, dict(workload=model, trial=trial, requests=requests, baseline=baseline)


def source_bundle():
    # Ship tracked source/config from the PC; no GitHub credentials needed on EC2.
    names = subprocess.check_output(['git','ls-files','-z','zkinfer','evaluation','experiments/config'],cwd=ROOT).decode().split('\0')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(filter(None,names)):
            p=ROOT/name
            if p.is_symlink(): raise ValueError('Source symlink unsupported: '+name)
            if p.is_file():
                entry=zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
                archive.writestr(entry,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
    data=buffer.getvalue()
    return data,hashlib.sha256(data).hexdigest()


def launcher_only_changes(old, new):
    def contents(data):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return {n:z.read(n) for n in z.namelist()}
    before,after=contents(old),contents(new)
    changed={n for n in before.keys()|after.keys() if before.get(n)!=after.get(n)}
    allowed={'evaluation/aws/control.py','evaluation/aws/coordinator.py',
             'evaluation/aws/remote.py','evaluation/aws/test_control.py','evaluation/aws/README.md',
             'evaluation/setup.md'}
    if changed-allowed:
        raise ValueError('Repair would change experiment/runtime inputs: '+str(sorted(changed-allowed)))
    return sorted(changed)


def shell_command(c, code):
    """Select the existing Conda installation; never invoke interactive conda activate."""
    candidates = [c['conda_executable']] if c['conda_executable'] else [
        '/home/ubuntu/miniconda3/bin/conda','/home/ubuntu/anaconda3/bin/conda',
        '/home/ubuntu/miniforge3/bin/conda','/opt/conda/bin/conda']
    encoded=base64.b64encode(code.encode()).decode()
    python_code='import base64; exec(base64.b64decode('+repr(encoded)+'))'
    return ('set -eu\nCONDA=""\nfor p in '+shlex.join(candidates)+
            '; do if [ -x "$p" ]; then CONDA="$p"; break; fi; done\n'
            '[ -n "$CONDA" ] || { echo "Set conda_executable in config.json"; exit 1; }\n'
            'runuser -u ubuntu -- env AWS_DEFAULT_REGION='+shlex.quote(c['region'])+
            ' "$CONDA" run --no-capture-output -n '+shlex.quote(c['conda_env'])+
            ' python -c '+shlex.quote(python_code))


class Controller:
    def __init__(self,c,profile):
        import boto3
        self.c=c
        self.session=boto3.Session(profile_name=profile,region_name=c['region'])
        self.ec2=self.session.client('ec2'); self.ssm=self.session.client('ssm')
        self.s3=self.session.client('s3'); self.iam=self.session.client('iam')
        self.prefix='evaluation/'+c['campaign']
        self.local=ROOT/'evaluation/results/aws-control'/c['campaign']
        self.local.mkdir(parents=True,exist_ok=True)

    def instances(self):
        pages=self.ec2.get_paginator('describe_instances').paginate(Filters=[
            {'Name':'tag:zkInferCampaign','Values':[self.c['campaign']]},
            {'Name':'instance-state-name','Values':['pending','running','stopping','stopped']}])
        result=[i for page in pages for r in page['Reservations'] for i in r['Instances']]
        for i in result:
            i['index']=int(next(t['Value'] for t in i['Tags'] if t['Key']=='zkInferIndex'))
        if len({i['index'] for i in result})!=len(result): raise ValueError('Duplicate worker indices')
        return sorted(result,key=lambda i:i['index'])

    def init(self):
        c=self.c
        role=c['instance_profile']
        trust={'Version':'2012-10-17','Statement':[{'Effect':'Allow','Principal':{'Service':'ec2.amazonaws.com'},'Action':'sts:AssumeRole'}]}
        try:
            existing=self.iam.get_role(RoleName=role)['Role']
            if existing['AssumeRolePolicyDocument']!=trust:
                raise ValueError('Existing role has different trust; choose a new dedicated instance_profile name')
        except self.iam.exceptions.NoSuchEntityException:
            self.iam.create_role(RoleName=role,AssumeRolePolicyDocument=json.dumps(trust),Description='zkInfer experiment worker: SSM and scoped S3')
        self.iam.attach_role_policy(RoleName=role,PolicyArn='arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore')
        policy={'Version':'2012-10-17','Statement':[
            {'Effect':'Allow','Action':['s3:GetBucketLocation'],'Resource':'arn:aws:s3:::'+c['bucket']},
            {'Effect':'Allow','Action':['s3:ListBucket'],'Resource':'arn:aws:s3:::'+c['bucket'],
             'Condition':{'StringLike':{'s3:prefix':['evaluation/*']}}},
            {'Effect':'Allow','Action':['s3:GetObject','s3:PutObject','s3:DeleteObject','s3:AbortMultipartUpload'],
             'Resource':'arn:aws:s3:::'+c['bucket']+'/evaluation/*'}]}
        self.iam.put_role_policy(RoleName=role,PolicyName='zkInferEvaluationS3',PolicyDocument=json.dumps(policy))
        try: ip=self.iam.get_instance_profile(InstanceProfileName=role)['InstanceProfile']
        except self.iam.exceptions.NoSuchEntityException:
            ip=self.iam.create_instance_profile(InstanceProfileName=role)['InstanceProfile']
        if ip['Roles'] and any(r['RoleName']!=role for r in ip['Roles']): raise ValueError('Instance profile already uses a different role')
        if not ip['Roles']: self.iam.add_role_to_instance_profile(InstanceProfileName=role,RoleName=role)
        # Add only the coordinator port from the SAME group; never expose it publicly.
        try:
            self.ec2.authorize_security_group_ingress(GroupId=c['security_group_id'],IpPermissions=[
                {'IpProtocol':'tcp','FromPort':c['port'],'ToPort':c['port'],
                 'UserIdGroupPairs':[{'GroupId':c['security_group_id']}]}])
        except Exception as exc:
            if getattr(exc,'response',{}).get('Error',{}).get('Code')!='InvalidPermission.Duplicate': raise
        print('Worker role/profile ready; TCP coordinator port allowed within the supplied group.')

    def launch(self,n):
        c=self.c
        group=self.ec2.describe_security_groups(GroupIds=[c['security_group_id']])['SecurityGroups'][0]
        if not c['subnet_id']:
            subs=self.ec2.describe_subnets(Filters=[{'Name':'vpc-id','Values':[group['VpcId']]},
                                                  {'Name':'default-for-az','Values':['true']}])['Subnets']
            if not subs: raise ValueError('No default subnet in security-group VPC. Set subnet_id explicitly.')
            c['subnet_id']=sorted(subs,key=lambda s:(s['AvailabilityZone'],s['SubnetId']))[0]['SubnetId']
            print('Selected default subnet:',c['subnet_id'],'(must have outbound HTTPS/S3 access)')
        subnet=self.ec2.describe_subnets(SubnetIds=[c['subnet_id']])['Subnets'][0]
        if subnet['VpcId']!=group['VpcId']: raise ValueError('Subnet and security group VPC differ')
        image=self.ec2.describe_images(ImageIds=[c['ami_id']])['Images'][0]
        if image['State']!='available': raise ValueError('Wait until AMI is available')
        existing={i['index']:i for i in self.instances()}
        for index in range(1,n+1):
            if index not in existing:
                # A stable idempotency token prevents duplicate launches after PC interruption.
                token=hashlib.sha256((c['campaign']+str(index)).encode()).hexdigest()
                result=self.ec2.run_instances(ImageId=c['ami_id'],InstanceType=c['instance_type'],MinCount=1,MaxCount=1,
                    ClientToken=token,IamInstanceProfile={'Name':c['instance_profile']},
                    MetadataOptions={'HttpTokens':'required'},
                    NetworkInterfaces=[{'DeviceIndex':0,'SubnetId':c['subnet_id'],
                        'Groups':[c['security_group_id']],'AssociatePublicIpAddress':c['associate_public_ip']}],
                    BlockDeviceMappings=[{'DeviceName':image['RootDeviceName'],'Ebs':{
                        'VolumeSize':c['root_volume_gib'],'VolumeType':'gp3','DeleteOnTermination':True,'Encrypted':True}}],
                    TagSpecifications=[{'ResourceType':'instance','Tags':[
                        {'Key':'Name','Value':c['campaign']+'-'+str(index)},
                        {'Key':'zkInferCampaign','Value':c['campaign']},{'Key':'zkInferIndex','Value':str(index)}]}],
                    UserData='#!/bin/bash\nset -eu\nif snap list amazon-ssm-agent >/dev/null 2>&1; then\n snap start --enable amazon-ssm-agent\nelif systemctl cat amazon-ssm-agent >/dev/null 2>&1; then\n systemctl enable --now amazon-ssm-agent\nelse\n snap install amazon-ssm-agent --classic\n snap start --enable amazon-ssm-agent\nfi\n')
                print('Launched',result['Instances'][0]['InstanceId'],'index',index,flush=True)
        active=[i for i in self.instances() if i['index']<=n]
        for i in active:
            if i['State']['Name']=='stopping': self.ec2.get_waiter('instance_stopped').wait(InstanceIds=[i['InstanceId']])
            if i['State']['Name'] in {'stopped','stopping'}: self.ec2.start_instances(InstanceIds=[i['InstanceId']])
        ids=[i['InstanceId'] for i in active]
        self.ec2.get_waiter('instance_status_ok').wait(InstanceIds=ids)
        until=time.monotonic()+600
        while time.monotonic()<until:
            managed=self.ssm.describe_instance_information(Filters=[{'Key':'InstanceIds','Values':ids}])['InstanceInformationList']
            if {i['InstanceId'] for i in managed if i['PingStatus']=='Online'}==set(ids): break
            time.sleep(10)
        else: raise TimeoutError('SSM not Online. Check agent, instance role and outbound HTTPS; see README.')
        return [i for i in self.instances() if i['index']<=n]

    def send(self,instance,script,timeout=600):
        result=self.ssm.send_command(InstanceIds=[instance],DocumentName='AWS-RunShellScript',
            Parameters={'commands':[script],'executionTimeout':[str(timeout)]},TimeoutSeconds=600,
            OutputS3BucketName=self.c['bucket'],OutputS3KeyPrefix=self.prefix+'/ssm')
        command=result['Command']['CommandId']
        print('SSM',instance,command,flush=True)
        return command

    def wait(self,instance,command,timeout):
        until=time.monotonic()+timeout+660
        while time.monotonic()<until:
            try: result=self.ssm.get_command_invocation(CommandId=command,InstanceId=instance)
            except self.ssm.exceptions.InvocationDoesNotExist:
                time.sleep(2); continue
            if result['Status']=='Success': return result
            if result['Status'] not in {'Pending','InProgress','Delayed','Cancelling'}:
                raise RuntimeError(json.dumps(result,default=str))
            time.sleep(5)
        raise TimeoutError('SSM polling deadline; inspect command '+command+' before restarting')

    def call(self,instance,action,timeout=600,**extra):
        payload={'config':self.c,'action':action,'instance_id':instance,**extra}
        encoded=base64.b64encode(json.dumps(payload).encode()).decode()
        code='import runpy,sys,os; os.chdir('+repr(self.c['repo'])+'); sys.argv=["remote.py",'+repr(encoded)+']; runpy.run_path("evaluation/aws/remote.py",run_name="__main__")'
        cmd=self.send(instance,shell_command(self.c,code),timeout)
        if action=='traffic':
            write(self.local/'traffic-command.json',{'instance':instance,'command':cmd,'timeout':timeout})
        return self.wait(instance,cmd,timeout)

    def install(self,instances,data,digest):
        key=self.prefix+'/source/'+digest+'.zip'
        self.s3.put_object(Bucket=self.c['bucket'],Key=key,Body=data)
        def one(instance):
            code=('import boto3,hashlib,io,zipfile,os\n'
                  'data=boto3.client("s3").get_object(Bucket='+repr(self.c['bucket'])+',Key='+repr(key)+')["Body"].read()\n'
                  'assert hashlib.sha256(data).hexdigest()=='+repr(digest)+'\n'
                  'os.chdir('+repr(self.c['repo'])+')\n'
                  'z=zipfile.ZipFile(io.BytesIO(data))\n'
                  'assert all(not n.startswith("/") and ".." not in n.split("/") for n in z.namelist())\n'
                  'z.extractall(".")\n')
            iid=instance['InstanceId']
            self.wait(iid,self.send(iid,shell_command(self.c,code)),600)
            self.call(iid,'doctor',source_sha256=digest)
        with ThreadPoolExecutor(max_workers=10) as pool: list(pool.map(one,instances))
        checks=[json.loads(self.s3.get_object(Bucket=self.c['bucket'],Key=self.prefix+'/checks/'+i['InstanceId']+'.json')['Body'].read()) for i in instances]
        for check in checks[1:]:
            for field in ('models','python','ezkl','ezkl_native_sha256','cpus','memory_gib','source_sha256'):
                if check[field]!=checks[0][field]: raise ValueError('Workers differ: '+field)

    def freeze(self,digest):
        key=self.prefix+'/campaign.json'
        value={'config':self.c,'source_sha256':digest}
        try: previous=json.loads(self.s3.get_object(Bucket=self.c['bucket'],Key=key)['Body'].read())
        except self.s3.exceptions.NoSuchKey: previous=None
        if previous and previous!=value: raise ValueError('Campaign settings/source changed. Use a new campaign name.')
        if not previous: self.s3.put_object(Bucket=self.c['bucket'],Key=key,Body=json.dumps(value).encode())
        write(self.local/'campaign.json',value)

    def run(self):
        if (self.local/'active.json').exists():
            raise RuntimeError('Previous trial needs recovery. Run recover --execute before resuming.')
        data,digest=source_bundle()
        # Resolve subnet before freezing (and never overwrite the user's config file).
        instances=self.launch(1)
        self.freeze(digest)
        self.install(instances,data,digest)
        leader=instances[0]['InstanceId']
        for model in self.c['workloads']:
            self.call(leader,'prepare',timeout=self.c['timeout_seconds']+600,workload=model)
        for n in self.c['nodes']:
            instances=self.launch(n)
            self.install(instances,data,digest)
            ids=[i['InstanceId'] for i in instances]
            leader=ids[0]
            ip=instances[0]['PrivateIpAddress']
            for key,cell in cells(self.c,n):
                donekey=self.prefix+'/completed/'+key+'.json'
                try:
                    self.s3.head_object(Bucket=self.c['bucket'],Key=donekey)
                    print('SKIP',key); continue
                except Exception as exc:
                    if getattr(exc,'response',{}).get('Error',{}).get('Code') not in {'404','NoSuchKey','NotFound'}: raise
                # Each retry has its own prefix. Never overwrite partial results.
                attempt=key+'/'+time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
                args=dict(attempt=attempt,coordinator_ip=ip,instance_ids=ids,**cell)
                write(self.local/'active.json',args)
                success=False
                try:
                    for iid in ids: self.call(iid,'init',leader=iid==leader,**args)
                    self.call(leader,'start',role='coordinator',**args)
                    for iid in ids: self.call(iid,'start',role='node',**args)
                    self.call(leader,'traffic',timeout=self.c['timeout_seconds']+300,**args)
                    success=True
                finally:
                    errors=[]
                    for iid in ids:
                        try: self.call(iid,'collect',timeout=3600,**args)
                        except Exception as exc: errors.append(str(exc))
                    if errors: raise RuntimeError('Collection incomplete; do not stop nodes yet: '+str(errors))
                if success:
                    self.s3.put_object(Bucket=self.c['bucket'],Key=donekey,Body=json.dumps(args).encode())
                    (self.local/'active.json').unlink(missing_ok=True)
                    (self.local/'traffic-command.json').unlink(missing_ok=True)
        self.stop()

    def recover(self):
        """Stop a previously interrupted trial and preserve it; never promote it to success."""
        path=self.local/'active.json'
        if not path.exists():
            print('No active trial recorded on this PC.'); return
        args=read(path)
        cmdpath=self.local/'traffic-command.json'
        if cmdpath.exists():
            cmd=read(cmdpath)
            result=self.ssm.get_command_invocation(CommandId=cmd['command'],InstanceId=cmd['instance'])
            if result['Status'] in {'Pending','InProgress','Delayed','Cancelling'}:
                self.ssm.cancel_command(CommandId=cmd['command'],InstanceIds=[cmd['instance']])
        for iid in args['instance_ids']:
            self.call(iid,'collect',timeout=3600,**args)
        write(self.local/('interrupted-'+uuid.uuid4().hex[:8]+'.json'),args)
        path.unlink(); cmdpath.unlink(missing_ok=True)
        print('Interrupted attempt retained; next run uses a fresh attempt. Instances remain running.')

    def repair_launcher(self):
        # A narrow migration for the initial Hydra startup failure. No completed
        # results or runtime/profile changes may be carried across this repair.
        if (self.local/'active.json').exists():
            raise RuntimeError('Run recover --execute first')
        key=self.prefix+'/campaign.json'
        old=json.loads(self.s3.get_object(Bucket=self.c['bucket'],Key=key)['Body'].read())
        config=dict(self.c)
        if not config['subnet_id']: config['subnet_id']=old['config']['subnet_id']
        if config!=old['config']: raise ValueError('Configuration differs; launcher repair cannot change settings')
        if self.s3.list_objects_v2(Bucket=self.c['bucket'],Prefix=self.prefix+'/completed/',MaxKeys=1).get('KeyCount',0):
            raise ValueError('Campaign has completed cells; use a new campaign instead')
        data,digest=source_bundle()
        old_data=self.s3.get_object(Bucket=self.c['bucket'],Key=self.prefix+'/source/'+old['source_sha256']+'.zip')['Body'].read()
        if hashlib.sha256(old_data).hexdigest()!=old['source_sha256']:
            raise ValueError('Stored source bundle hash mismatch')
        changed=launcher_only_changes(old_data,data)
        self.s3.put_object(Bucket=self.c['bucket'],Key=self.prefix+'/source/'+digest+'.zip',Body=data)
        record={'previous':old,'changed_files':changed,'new_source_sha256':digest}
        self.s3.put_object(Bucket=self.c['bucket'],Key=self.prefix+'/repairs/'+uuid.uuid4().hex+'.json',Body=json.dumps(record).encode())
        updated={'config':config,'source_sha256':digest}
        self.s3.put_object(Bucket=self.c['bucket'],Key=key,Body=json.dumps(updated).encode())
        write(self.local/'campaign.json',updated)
        print('Launcher repaired; retained existing AWS profiles and setup. Run run --execute next.')

    def stop(self):
        ids=[i['InstanceId'] for i in self.instances() if i['State']['Name'] in {'pending','running'}]
        if ids: self.ec2.stop_instances(InstanceIds=ids)
        print('Stopped campaign instances; EBS volumes/results retained:',ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','launch','run','status','recover','repair-launcher','stop'])
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--profile',help='AWS CLI profile name; omit for default credential chain')
    p.add_argument('--nodes',type=int,default=1)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); c=read(a.config); validate(c)
    if not 1<=a.nodes<=10: p.error('--nodes must be in 1..10')
    if not a.execute:
        print(json.dumps(c,indent=2))
        print('PLAN ONLY:',a.action,'No AWS calls made.')
        if a.action=='run':
            for n in c['nodes']:
                for key,_ in cells(c,n): print(key)
        return
    controller=Controller(c,a.profile)
    if a.action=='init': controller.init()
    elif a.action=='launch': print(json.dumps(controller.launch(a.nodes),default=str,indent=2))
    elif a.action=='status': print(json.dumps(controller.instances(),default=str,indent=2))
    elif a.action=='stop': controller.stop()
    elif a.action=='recover': controller.recover()
    elif a.action=='repair-launcher': controller.repair_launcher()
    else: controller.run()


if __name__=='__main__': main()
