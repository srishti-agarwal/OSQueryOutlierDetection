from collections import defaultdict
def fillBasicFeatures(event, feature, bdict):
    fd = getattr(event, "columns")

    if feature == "scheduled_tasks":
        bdict[feature].append(fd["name"])
    if feature == "shared_resources":
        bdict[feature].append(fd["name"])
    if feature == "wmi_cli_event_consumers":
        bdict[feature].append(fd["name"])
    if feature == "logged_in_users":
        bdict[feature].append(fd["user"])
    if feature == "services":
        bdict[feature].append(fd["name"])
    if feature == "etc_hosts":
        bdict[feature].append(fd["name"])
    return bdict

def fillNetworkFeatures(event, feature, ndict):
    fd = getattr(event, "columns")
    if feature is "open_ports":
        ndict[feature].append(fd["port"])
    return ndict

def fillRegistryFeatures(event, feature, regdict):
    fd = getattr(event, "columns")
    if feature is "startup_items":
        regdict[feature].append(fd["name"])
    return regdict

def fillWindowsFeatures(event,feature,windict):
    fd = getattr(event, "columns")
    windict[feature].append(fd["name"])
    return windict


def aggregateFeature(rowindex, dataframe, fdict):
    for key, value in fdict.items():
        if key == "percent_disk_time":
            dataframe.at[rowindex, key] = max(fdict[key])
        else:
            dataframe.at[rowindex, key] = len(set(fdict[key]))

