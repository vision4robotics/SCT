# echo "evaluating UAVDT"
# echo "offline running"
# echo "dimp50"
# python run_tracker.py --tracker_name dimp --tracker_param dimp50 --dataset_name uavdt
# echo "dimp18"
# python run_tracker.py --tracker_name dimp --tracker_param dimp18 --dataset_name uavdt
# echo "prdimp18"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp18 --dataset_name uavdt

# echo "prdimp50"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp50 --dataset_name uavdt

# echo "ATOM"
# python run_tracker.py --tracker_name atom --tracker_param default --dataset_name uavdt

# echo "evaluating DTB70"
# echo "offline running"
# echo "dimp50"
# python run_tracker.py --tracker_name dimp --tracker_param dimp50 --dataset_name dtb70
# echo "dimp18"
# python run_tracker.py --tracker_name dimp --tracker_param dimp18 --dataset_name dtb70
# echo "prdimp18"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp18 --dataset_name dtb70

# echo "prdimp50"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp50 --dataset_name dtb70

# echo "ATOM"
# python run_tracker.py --tracker_name atom --tracker_param default --dataset_name dtb70

# echo "evaluating visdrone"
# echo "offline running"
# echo "dimp50"
# python run_tracker.py --tracker_name dimp --tracker_param dimp50 --dataset_name visdrone
# echo "dimp18"
# python run_tracker.py --tracker_name dimp --tracker_param dimp18 --dataset_name visdrone
# echo "prdimp18"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp18 --dataset_name visdrone

# echo "prdimp50"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp50 --dataset_name visdrone

# echo "ATOM"
# python run_tracker.py --tracker_name atom --tracker_param default --dataset_name visdrone

echo "evaluating DarkTrack"
echo "offline running"
echo "dimp18"
python run_tracker.py --tracker_name dimp --tracker_param dimp18 --enhance
echo "dimp50"
python run_tracker.py --tracker_name dimp --tracker_param dimp50 --enhance
echo "prdimp50"
python run_tracker.py --tracker_name dimp --tracker_param prdimp50 --enhance

# echo "ATOM"
# python run_tracker.py --tracker_name atom --tracker_param default --enhance True


# echo "dimp18"
# python run_tracker.py --tracker_name dimp --tracker_param dimp18 --dataset_name uav10fps
# echo "prdimp18"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp18 --dataset_name uav10fps

# echo "prdimp50"
# python run_tracker.py --tracker_name dimp --tracker_param prdimp50 --dataset_name uav10fps

# echo "ATOM"
# python run_tracker.py --tracker_name atom --tracker_param default --dataset_name uav10fps