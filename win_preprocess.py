import pandas as pd
import os

from collections import defaultdict
from features import fillBasicFeatures, fillNetworkFeatures, fillRegistryFeatures, fillWindowsFeatures, aggregateFeature
from OCSVM import applySVM
from PCA import applyPCAModel
basicfeatureList = ["scheduled_tasks", "shared_resources", "wmi_cli_event_consumers",
                                    "logged_in_users","services", "etc_hosts"]
networkList = ["open_ports"]
registryList = ["startup_items"]
windowAttackList =["pack_windows-attacks_CCleaner_Trojan.Floxif", "Winsecurity_info_1","unTabs_1", "conhost.exe_incorrect_path",
                      "dllhost.exe_incorrect_path", "lsass.exe_incorrect_path","services.exe_incorrect_parent_process",
                      "svchost.exe_incorrect_path", "svchost.exe_incorrect_parent_process","wmiprvse.exe_incorrect_path",
                      "winlogon.exe_incorrect_path"]
finalFeatures = ["eventTime"]+[v for v in basicfeatureList+networkList+registryList+windowAttackList]
dataframe_train = pd.DataFrame(columns=finalFeatures)
dataframe_test = pd.DataFrame(columns=finalFeatures)

def processLogs(dataframe,logpath):
    bdict = defaultdict(list)
    ndict = defaultdict(list)
    regdict = defaultdict(list)
    windict = defaultdict(list)
    path = os.path.join('logs',logpath)
    print (path)
    data = pd.read_json(path, lines=True)
    # data = pd.read_json('logs/logs.txt', lines=True)
    data['calendarTime'] = data['calendarTime'].apply(lambda x: x[0:x.rfind(':')])
    group_using_time = data.groupby(['unixTime'])
    eventList = list(group_using_time)
    rowIndex = 0
    for timeLine in eventList:
        dataframe.at[rowIndex,"eventTime"] = timeLine[0]
        series = timeLine[1]
        for row in series.itertuples():
            feature = getattr(row, "name")
            if feature in basicfeatureList:
                fillBasicFeatures(row, feature, bdict)
            elif feature in networkList:
                fillNetworkFeatures(row, feature, ndict)
            elif feature in registryList:
                fillRegistryFeatures(row,feature,regdict)
            elif feature in windowAttackList:
                fillWindowsFeatures(row, feature,windict)
        aggregateFeature(rowIndex, dataframe, dict(bdict,**ndict,**regdict,**windict))
        bdict.clear(), ndict.clear(), regdict.clear(), windict.clear()
        rowIndex +=1


processLogs(dataframe_train,'osqueryd_results_train.log')
processLogs(dataframe_test,'osqueryd_results_test.log')
dataframe_train = dataframe_train.fillna(0)
dataframe_test= dataframe_test.fillna(0)
# applyPCAModel(dataframe_train,dataframe_test)
applySVM(dataframe_train, dataframe_test)

