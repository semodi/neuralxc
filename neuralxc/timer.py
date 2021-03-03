import time

import pandas as pd
from tabulate import tabulate

import neuralxc.config as config


class DummyTimer():
    def start(self, name, *args, **kwargs):
        pass

    def stop(self, stop, *args, **kwargs):
        pass

    def create_report(self, path, *args, **kwargs):
        pass


class Timer():
    def __init__(self):
        print("NEURALXC: Timer started")
        self.start_dict = {'master': time.time()}
        self.cnt_dict = {'master': 1}
        self.accum_dict = {}
        self.max_dict = {}
        self.min_dict = {}
        self.path = 'NXC_TIMING'
        self.threaded = False

    def start(self, name, threadsafe=True):

        if not (self.threaded and not threadsafe):
            if name in self.cnt_dict:
                self.cnt_dict[name] += 1
            else:
                self.cnt_dict[name] = 1

            if not name in self.start_dict:
                self.start_dict[name] = time.time()

    def stop(self, name, threadsafe=True):

        if not (self.threaded and not threadsafe):
            if name in self.start_dict:
                dt = time.time() - self.start_dict[name]
                if name in self.accum_dict:
                    self.accum_dict[name] += time.time() - self.start_dict[name]
                    self.max_dict[name] = max(self.max_dict[name], dt)
                    self.min_dict[name] = min(self.min_dict[name], dt)
                else:
                    self.accum_dict[name] = time.time() - self.start_dict[name]
                    self.max_dict[name] = dt
                    self.min_dict[name] = dt

                self.start_dict.pop(name)
            else:
                raise ValueError('Timer with name {} was never started'.format(name))

    def create_report(self, path=None):
        keys = list(self.start_dict.keys())
        # for key in keys:
        #     if not key in self.accum_dict:
        #         self.stop(key)

        report = pd.DataFrame.from_dict(self.accum_dict, orient='index', columns=['Total time'])
        report['Calls'] = pd.DataFrame.from_dict(self.cnt_dict, orient='index', columns=['Calls'])
        report['Time per call'] = report['Total time'] / report['Calls']
        report['Best'] = pd.DataFrame.from_dict(self.min_dict, orient='index', columns=['Best'])
        report['Worst'] = pd.DataFrame.from_dict(self.max_dict, orient='index', columns=['Worst'])
        report["% of master"] = report['Total time'] / report['Total time'].loc['master'] * 100
        if path:
            open(path, 'w').write(tabulate(report, tablefmt="pipe", headers="keys"))
        else:
            print(report)


if config.UseTimer:
    timer = Timer()
else:
    timer = DummyTimer()
