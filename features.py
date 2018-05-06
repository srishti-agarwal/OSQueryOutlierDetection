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
        bdict[feature].append(fd["host"])
    if feature == "services":
        bdict[feature].append(fd["name"])
    if feature == "etc_hosts":
        bdict[feature].append(fd["name"])
    if feature == "svchost.exe_incorrect_path":
        bdict[feature].append(fd["path"])
    return bdict

def fillNetworkFeatures(event, feature, ndict):
    fd = getattr(event, "columns")
    if feature == "open_ports":
        ndict[feature].append(fd['port'])
    return ndict

def fillRegistryFeatures(event, feature, regdict):
    fd = getattr(event, "columns")
    if feature == "startup_items":
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

featureList = ["pack_user_behavior_usb_devices", "pack_user_behavior_kernel_modules", "pack_user_behavior_open_sockets",
               "pack_user_behavior_open_files","pack_user_behavior_logged_in_users", "pack_user_behavior_shell_history",
               "pack_user_behavior_listening_ports", "pack_user_behavior_arp_cache", "pack_user_behavior_processes"]

def fillLinuxFeatures(event, feature, flist):
    fd = getattr(event, "columns")
    if feature == "pack_user_behavior_usb_devices":
        flist[feature].append(fd["name"])
    if feature == "pack_user_behavior_kernel_modules":
        flist[feature].append(fd["name"])
    if feature == "pack_user_behavior_open_sockets":
        flist[feature].append(fd["remote_address"])
    if feature == "pack_user_behavior_open_files":
        flist[feature].append(fd["path"])
    if feature == "pack_user_behavior_logged_in_users":
        flist[feature].append(fd["tty"])
    if feature == "pack_user_behavior_shell_history":
        flist[feature].append(fd["command"])
    if feature == "pack_user_behavior_listening_ports":
        flist[feature].append(fd["port"])
    if feature == "pack_user_behavior_arp_cache":
        flist[feature].append(fd["address"])
    if feature == "pack_user_behavior_processes":
        flist[feature].append(fd["name"])
    return flist






