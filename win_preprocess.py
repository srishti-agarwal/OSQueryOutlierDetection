import pandas as pd
import os
import custom_function as cf

from collections import defaultdict
from features import fillBasicFeatures, fillNetworkFeatures, fillRegistryFeatures, fillWindowsFeatures, aggregateFeature

class processData():
    def __init__(self):
        self.basicfeatureList = ["scheduled_tasks", "shared_resources", "wmi_cli_event_consumers",
                                    "logged_in_users","services", "etc_hosts", "svchost.exe_incorrect_path"]
        self.networkList = ["open_ports"]
        self.registryList = ["startup_items"]
        self.windowAttackList =["pack_windows-attacks_CCleaner_Trojan.Floxif", "pack_Winsecurity_info_1","pack_unTabs_1", "pack_conhost.exe_incorrect_path",
                      "pack_dllhost.exe_incorrect_path", "pack_lsass.exe_incorrect_path","pack_services.exe_incorrect_parent_process",
                      "pack_svchost.exe_incorrect_path", "pack_svchost.exe_incorrect_parent_process","pack_wmiprvse.exe_incorrect_path",
                      "pack_winlogon.exe_incorrect_path"]
        self.finalFeatures = ["eventTime"]+[v for v in self.basicfeatureList+self.networkList+self.registryList+self.windowAttackList]
        self.dataframe = pd.DataFrame(columns=self.finalFeatures)


    def processLogs(self,logpath):
        bdict = defaultdict(list)
        ndict = defaultdict(list)
        regdict = defaultdict(list)
        windict = defaultdict(list)
        path = os.path.join('data',logpath)
        print (path)
        data = pd.read_json(path, lines=True)
        data['calendarTime'] = data['calendarTime'].apply(lambda x: x[0:x.rfind(':')])
        group_using_time = data.groupby(['calendarTime'])
        eventList = list(group_using_time)
        rowIndex = 0
        for timeLine in eventList:
            self.dataframe.at[rowIndex,"eventTime"] = timeLine[0]
            series = timeLine[1]
            for row in series.itertuples():
                feature = getattr(row, "name")
                if feature in self.basicfeatureList:
                    fillBasicFeatures(row, feature, bdict)
                if feature in self.networkList:
                    fillNetworkFeatures(row, feature, ndict)
                if feature in self.registryList:
                    fillRegistryFeatures(row,feature,regdict)
                if feature in self.windowAttackList:
                    fillWindowsFeatures(row, feature,windict)
            fullDict = cf.merge_two_dicts(bdict,cf.merge_two_dicts(cf.merge_two_dicts(ndict,regdict),windict))
            aggregateFeature(rowIndex, self.dataframe, fullDict)
            bdict.clear(), ndict.clear(), regdict.clear(), windict.clear()
            rowIndex +=1