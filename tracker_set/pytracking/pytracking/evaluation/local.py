from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/ye/Documents/SOT/trackers/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = ''
    settings.results_path = '/home/ye/Documents/My_work/0_remote/SCT/tracker_set/pytracking/results/'    # Where to store tracking results
    settings.results_path_rt = ''
    settings.segmentation_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = '/media/v4r/My_Passport/dataset/UAV123/'
    settings.uavdark_path = '/media/ye/My_Passport/dataset/UAVDark135_TSP_out/' 
    settings.darktrack_path = '/media/ye/Luck/dataset/DarkTrack2021/'
    settings.uav20l_path = '/media/v4r/Luck/dataset/UAV20L'
    settings.uav10fps_path = '/media/v4r/Luck/dataset/UAV123_10fps'
    settings.youtubevos_dir = ''

    return settings

