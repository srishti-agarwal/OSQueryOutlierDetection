import pandas as pd
import os

from collections import defaultdict
from features import fillBasicFeatures, fillNetworkFeatures, fillRegistryFeatures, fillWindowsFeatures, aggregateFeature, fillLinuxFeatures
featureList = ["pack_user_behavior_usb_devices", "pack_user_behavior_kernel_modules", "pack_user_behavior_open_sockets",
               "pack_user_behavior_open_files","pack_user_behavior_logged_in_users", "pack_user_behavior_shell_history",
               "pack_user_behavior_listening_ports", "pack_user_behavior_arp_cache", "pack_user_behavior_processes"]

finalFeatures = ["eventTime"]+[v for v in featureList]
dataframe_train = pd.DataFrame(columns=finalFeatures)
dataframe_test = pd.DataFrame(columns=finalFeatures)

def processLogs(dataframe,logpath):
    fdict = defaultdict(list)
    path = os.path.join('logs',logpath)
    print (path)
    data = pd.read_json(path, lines=True)
    # data = pd.read_json('logs/logs.txt', lines=True)
    data['calendarTime'] = data['calendarTime'].apply(lambda x: x[0:x.rfind(':')])
    group_using_time = data.groupby(['unixTime'])
    eventList = list(group_using_time)
    # print eventList[:3]
    rowIndex = 0
    for timeLine in eventList:
        print "timeLine is : ", timeLine[0]
        dataframe.at[rowIndex,"eventTime"] = timeLine[0]
        series = timeLine[1]
        # print "series is: ", series
        for row in series.itertuples():
            feature = getattr(row, "name")
            # print "feature is: ", feature
            if feature in featureList:
                fillLinuxFeatures(row, feature, fdict)
        # print "fdict is: ", fdict
        aggregateFeature(rowIndex, dataframe, dict(fdict))
        # print "Building dt: ", dataframe
        fdict.clear()
        rowIndex +=1