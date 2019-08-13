import time
import pandas as pd
from tabulate import tabulate

class DummyTimer():


    def start(self,name):
        pass
    def stop(self,stop):
        pass

    def create_report(self,path):
        pass
class Timer():

    def __init__(self):
        print("NEURALXC: Timer started")
        self.start_dict = {'master' : time.time()}
        self.cnt_dict = {'master' : 1}
        self.accum_dict = {}
        self.path = 'NXC_TIMING'

    def start(self, name):
        if name in self.cnt_dict:
            self.cnt_dict[name] +=1
        else:
            self.cnt_dict[name] = 1

        if not name in self.start_dict:
            self.start_dict[name] = time.time()



    def stop(self, name):
        if name in self.start_dict:
            if name in self.accum_dict:
                self.accum_dict[name] += time.time() - self.start_dict[name]
            else:
                self.accum_dict[name] = time.time() - self.start_dict[name]

            self.start_dict.pop(name)
        else:
            raise ValueError('Timer with name {} was never started'.format(name))

    def create_report(self,path = None):
        keys = list(self.start_dict.keys())
        for key in keys:
            if not key in self.accum_dict:
                self.stop(key)

        report = pd.DataFrame.from_dict(self.accum_dict, orient='index', columns = ['Total time'])
        report['Calls']= pd.DataFrame.from_dict(self.cnt_dict, orient='index', columns = ['Calls'])
        report['Time per call'] = report['Total time']/report['Calls']
        report["% of master" ] = report['Total time']/report['Total time'].loc['master'] * 100
        if path:
            open(path,'w').write(tabulate(report, tablefmt="pipe", headers="keys"))
        else:
            print(report)

#timer = Timer()
timer = DummyTimer()
