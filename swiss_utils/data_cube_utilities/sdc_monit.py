# Copyright 2018 GRID-Geneva. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# Import necessary stuff
import time
import datetime
import psutil
from IPython.display import clear_output # , IFrame

def monit_sys(proc_time = 10):
    """
    Description:
      Monitor and average CPU and RAM activity during a given time
    -----
    Input:
      proc_time: monitoring time (in seconds, 10 by default)
    Output:
      on screen
    """
    
    blocks_nb = 20 # length of the percentage bar
    cpu_log = []
    mem_log = []

    for i in range(proc_time, 0, -1):
        # get used CPU percentage values
        cpu_pc = psutil.cpu_percent()
        cpu_blocks = int(cpu_pc / 100 * blocks_nb)
        cpu_log.append(cpu_pc)
        
        # get used RAM percentage
        mem_pc = psutil.virtual_memory().percent
        mem_blocks = int(mem_pc / 100 * blocks_nb)
        mem_log.append(mem_pc)

        # Print out instant values
        clear_output(wait = True) # refresh display
        print('Monitoring: wait %i seconds' % (i))
        print('CPU\t[%s%s]' % ('#' * cpu_blocks, '-' * (blocks_nb - cpu_blocks)))
        print('MEM\t[%s%s]' % ('#' * mem_blocks, '-' * (blocks_nb - mem_blocks)))

        time.sleep(1)
    
    # Calulate average
    cpu_avg = sum(cpu_log) / len(cpu_log)
    mem_avg = sum(mem_log) / len(mem_log)
    
    # Print out averaged values
    clear_output(wait = True)
    print('%i seconds average:' % (proc_time))
    print('CPU\t %5.1f%%' % (cpu_avg))
    print('MEM\t %5.1f%%' % (mem_avg))

def activity_logger(log_name = 'activity.log', interval_s = 10):
    """
    Description:
      Loggin RAM, Swap and CPU activity at a given interval until kernel is interrupted
    -----
    Input:
      log_name: (OPTIONAL) name of log file to create  (activity.log by default)
      interval_s: (OPTIONAL) loggin interval in seconds (10 seconds per default)
    Output:
      log file
    """

    f = open(log_name,'w')
    f.write('date,hour,ram_used_Mb,swap_used_Mb,cpu_pc\n')

    print('Logging activity until kernel is interrupted')
    try:
        while True:
            currentDT = datetime.datetime.now()
            ram_used = (psutil.virtual_memory().total - psutil.virtual_memory().available)
            swap_used = psutil.swap_memory().used
            cpu_pc = psutil.cpu_percent()
            f.write('%s,%s,%s,%s\n' %
                    (currentDT.strftime("%d/%m/%Y,%H:%M:%S"),
                     int(ram_used / 1024 / 1024),
                     int(swap_used / 1024 / 1024),
                     round(cpu_pc, 1)))
            time.sleep(interval_s)
    except KeyboardInterrupt:
        pass

    f.close()

    return 0
