import pandas as pd
import os
from collections import defaultdict
from features import aggregateFeature, fillLinuxFeatures

class processLData():
    def __init__(self):
        self.featureList = ["pack_user_behavior_usb_devices", "pack_user_behavior_kernel_modules", "pack_user_behavior_open_sockets",
               "pack_user_behavior_open_files","pack_user_behavior_logged_in_users", "pack_user_behavior_shell_history",
               "pack_user_behavior_listening_ports", "pack_user_behavior_arp_cache", "pack_user_behavior_processes"]

        self.finalFeatures = ["eventTime"]+[v for v in self.featureList]
        self.dataframe = pd.DataFrame(columns=self.finalFeatures)

    def processLogs(self,logpath):
        fdict = defaultdict(list)
        path = os.path.join('data',logpath)
        print (path)
        data = pd.read_json(path, lines=True)
        data['calendarTime'] = data['calendarTime'].apply(lambda x: x[0:x.rfind(':')])
        #group all the features by time.
        group_using_time = data.groupby(['unixTime'])
        eventList = list(group_using_time)
        rowIndex = 0
        for timeLine in eventList:
            self.dataframe.at[rowIndex,"eventTime"] = timeLine[0]
            series = timeLine[1]
            # print "series is: ", series
            for row in series.itertuples():
                feature = getattr(row, "name")
                # print "feature is: ", feature
                if feature in self.featureList:
                    fillLinuxFeatures(row, feature, fdict)
            # print "fdict is: ", fdict
            aggregateFeature(rowIndex, self.dataframe, dict(fdict))
            # print "Building dt: ", dataframe
            fdict.clear()
            rowIndex +=1