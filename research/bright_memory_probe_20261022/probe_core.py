"""Independent integer-buffer checks; no BRIGHT/model/dataset imports."""
import hashlib
import json
import time

import numpy as np

PATTERNS=('zero','ones','alternating_a','alternating_5','address','weight_bits','walking_bit')


def pattern(kind,start,count,cycle):
    constants={'zero':0,'ones':0xffffffff,'alternating_a':0xaaaaaaaa,
               'alternating_5':0x55555555,'weight_bits':0x3dd5a746}
    if kind in constants:
        result=np.full(count,constants[kind],dtype=np.uint32)
    elif kind=='walking_bit':
        shifts=(np.arange(start,start+count,dtype=np.uint32)+np.uint32(cycle%32))%np.uint32(32)
        result=np.left_shift(np.uint32(1),shifts)
    elif kind=='address':
        result=np.arange(start,start+count,dtype=np.uint32)
        result ^= np.uint32((cycle*0x9e3779b9)&0xffffffff)
        result ^= result >> np.uint32(16)
        result *= np.uint32(0x7feb352d)
        result ^= result >> np.uint32(15)
        result *= np.uint32(0x846ca68b)
        result ^= result >> np.uint32(16)
    else:raise ValueError('Unknown pattern: '+kind)
    return result.view(np.int32)


def differences(expected,actual,offset):
    e=expected.view(np.uint32);a=actual.view(np.uint32)
    indices=np.flatnonzero(e!=a)
    samples=[]
    for index in indices[:16]:
        old,new=int(e[index]),int(a[index]);xor=old^new
        samples.append(dict(word_index=offset+int(index),byte_offset=4*(offset+int(index)),
            expected_hex=hex(old),actual_hex=hex(new),xor_hex=hex(xor),changed_bits=xor.bit_count()))
    return dict(changed_words=int(indices.size),samples=samples)


class BufferMismatch(RuntimeError):
    def __init__(self,record):
        super().__init__('Independent buffer mismatch at '+record['phase'])
        self.record=record


def check_chunk(torch,a,b,expected,*,offset,phase,folder,position):
    # The reference is CPU-generated, rather than just trusting two GPU copies.
    reference_hash=hashlib.sha256(expected.tobytes()).hexdigest()
    gpu_equal=bool(torch.equal(a,b))
    first_a=a.detach().to('cpu',copy=True).numpy()
    first_b=b.detach().to('cpu',copy=True).numpy()
    da,db=differences(expected,first_a,offset),differences(expected,first_b,offset)
    reference_unchanged=reference_hash==hashlib.sha256(expected.tobytes()).hexdigest()
    if gpu_equal and not da['changed_words'] and not db['changed_words'] and reference_unchanged:return
    record=dict(position=dict(position),phase=phase,offset=offset,gpu_a_equals_b=gpu_equal,
        cpu_expected_unchanged=reference_unchanged,a_vs_expected=da,b_vs_expected=db,
        reference_sha256=reference_hash,hardware_fault_proven=False)
    # Preserve the original observations before trying any further GPU read.
    np.savez_compressed(folder/'first_mismatch_cpu.npz',expected=expected,a=first_a,b=first_b)
    (folder/'mismatch.json').write_text(json.dumps(record,indent=2))
    try:
        second_a=a.detach().to('cpu',copy=True).numpy()
        second_b=b.detach().to('cpu',copy=True).numpy()
        record['reread_a_vs_expected']=differences(expected,second_a,offset)
        record['reread_b_vs_expected']=differences(expected,second_b,offset)
        record['reread_a_same_as_first']=bool(np.array_equal(first_a,second_a))
        record['reread_b_same_as_first']=bool(np.array_equal(first_b,second_b))
        np.savez_compressed(folder/'reread_cpu.npz',a=second_a,b=second_b)
    except Exception as exc:
        record['reread_error']=type(exc).__name__+': '+str(exc)
    (folder/'mismatch.json').write_text(json.dumps(record,indent=2))
    raise BufferMismatch(record)


def exercise(torch,*,seconds,mib,folder,stop,emit,device='cuda:0',compute_seconds=3):
    """Bounded rounds; reports no-mismatch only, never certifies hardware."""
    words=mib*1024*1024//8  # two buffers; MiB is their combined size
    chunk_words=min(words,4*1024*1024)  # <=16MiB per CPU transfer
    a=torch.empty(words,dtype=torch.int32,device=device)
    b=torch.empty_like(a)
    # Exact all-ones GEMM stresses a different GPU operation between reads.
    side=2048 if str(device).startswith('cuda') else 16
    left=torch.ones((side,side),device=device,dtype=torch.float32)
    right=torch.ones_like(left);product=torch.empty_like(left)
    started=time.monotonic();deadline=started+seconds;completed=[];cycle=0
    position={};checked=0
    def boundary():
        if stop['requested']:raise InterruptedError('Stop requested')
    while time.monotonic()<deadline:
        boundary();kind=PATTERNS[cycle%len(PATTERNS)]
        position.update(cycle=cycle,pattern=kind,phase='write')
        emit(dict(position=position.copy(),completed_rounds=len(completed),checked_bytes=checked))
        for offset in range(0,words,chunk_words):
            boundary();count=min(chunk_words,words-offset)
            source=torch.from_numpy(pattern(kind,offset,count,cycle))
            a[offset:offset+count].copy_(source,non_blocking=False)
            b[offset:offset+count].copy_(a[offset:offset+count],non_blocking=False)
        for phase in ('after_write','after_compute'):
            position['phase']=phase
            if phase=='after_compute':
                compute_end=min(deadline,time.monotonic()+compute_seconds)
                while time.monotonic()<compute_end:
                    boundary();torch.mm(left,right,out=product)
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                # At least one multiply, including if the time budget expired.
                torch.mm(left,right,out=product)
                if not bool((product==float(side)).all()):
                    raise RuntimeError('Independent all-ones matrix multiplication produced an incorrect result')
            for offset in range(0,words,chunk_words):
                boundary();count=min(chunk_words,words-offset)
                check_chunk(torch,a[offset:offset+count],b[offset:offset+count],
                    pattern(kind,offset,count,cycle),offset=offset,phase=phase,folder=folder,position=position)
                checked+=count*8
        completed.append(kind);cycle+=1
        emit(dict(position=position.copy(),completed_rounds=len(completed),checked_bytes=checked))
    return dict(completed_rounds=len(completed),patterns_completed=sorted(set(completed)),
        all_patterns_completed=set(completed)==set(PATTERNS),checked_bytes=checked,
        allocated_test_buffer_mib=mib,elapsed_seconds=round(time.monotonic()-started,2),
        last_position=position,hardware_fault_proven=False,hardware_cleared=False)
