"""Measured concurrency selection for useful, unchanged MedSAM experiments."""
import math


class Capacity:
    def __init__(self, window=180, maximum=0):
        self.window = window
        self.maximum = maximum
        self.target = 1
        self.allowance = 12.
        self.peak = 0.
        self.samples = []
        self.signature = None
        self.measurements = {}
        self.decision = None
        self.events = []

    def limit(self, gpu):
        result = min(gpu['cpu_slots'], max(1, int((gpu['total_gib']-4)/self.allowance)))
        return min(result, self.maximum) if self.maximum else result

    def admission(self, gpu, workers):
        reservation = sum(max(0, self.allowance-gpu['processes'].get(w['pid'], 0)) for w in workers)
        return (len(workers) < min(self.target, self.limit(gpu))
                and gpu['total_gib']-gpu['used_gib'] >= 4+self.allowance+reservation
                and gpu['host_free_gib'] >= 10+12*(1+sum(w['step'] < 10 for w in workers))
                and all(w['step'] >= 10 or w['phase'] == 'evaluation' for w in workers))

    def observe(self, now, gpu, workers):
        trained = [w.get('peak_gib', 0) for w in workers if w['step'] >= 10]
        if trained and max(trained) > 0:
            # nvidia-smi includes CUDA context and allocator reservations.
            self.peak = max(self.peak,max(trained))
            self.allowance = max(11., 1.10*self.peak+.25)
        if self.target > self.limit(gpu):
            self.target = self.limit(gpu)
        signature = tuple((w['name'],w['phase']) for w in workers)
        if signature != self.signature:
            self.samples = []
            self.signature = signature
        if self.decision or len(workers) != self.target or not workers or any(
                w['phase'] != 'training' or w['step'] < 10 for w in workers):
            self.samples = []
            return
        self.samples.append((now, sum(w['step'] for w in workers),gpu['utilization']))
        first = self.samples[0]
        if now-first[0] < self.window:
            return
        rate = 60*(self.samples[-1][1]-first[1])/(now-first[0])
        record = dict(workers=self.target, updates_per_minute=rate, seconds=now-first[0],
                      updates=self.samples[-1][1]-first[1], gpu_utilization=sum(x[2] for x in self.samples)/len(self.samples),
                      memory_allowance_gib=self.allowance, resource_ceiling=self.limit(gpu))
        self.measurements.setdefault(self.target, []).append(record)
        self.events.append(record)
        self.samples = []
        # Two complete windows at each multi-worker level reduce checkpoint
        # quantization noise. All updates belong to real pending experiments.
        if len(self.measurements[self.target]) < (1 if self.target == 1 else 2):
            return
        rates = {n:sum(r['updates'] for r in rows)/sum(r['seconds'] for r in rows)*60
                 for n,rows in self.measurements.items()}
        if self.target < self.limit(gpu):
            self.target += 1
        else:
            eligible = [n for n in rates if n<=self.limit(gpu) and (n==1 or rates[n]>=1.08*rates[1])]
            self.target = max(eligible,key=lambda n:rates[n])
            self.decision = dict(workers=self.target, reason='Best measured throughput across feasible worker counts; extra workers require >=8% gain over one', rates=rates)

    def state(self):
        return dict(target_workers=self.target, per_worker_allowance_gib=self.allowance,
                    measurements=self.measurements, decision=self.decision,
                    criterion='Use extra workers only with >=8% aggregate training throughput gain; 180-second windows, two per multi-worker level')
