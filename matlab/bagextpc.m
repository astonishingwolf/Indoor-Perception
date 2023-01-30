files = dir(['C:\Users\dasgu\Documents\GithubPercep\sensordata\Sept_Recording\Sep28-2022_CAM-LCR_LIDAR_applanix\14-50-47\*.bag']); % Enter the bag file location here
n = length(files);
%print('hey')
for k = 1:n
    bagselect = rosbag(files(k).name);
    bSel = select(bagselect,'Topic','/rslidar_points_front');
    msgStructs = readMessages(bSel,'DataFormat','struct');
    pt = cell(length(msgStructs),1);
    for i = 1:length(msgStructs)
        img = rosReadXYZ(msgStructs{i,1});
        pt{i} = img;
    end
    save(append("pcdata/2022-06-16-13-01-40_",string(k),"_pc.mat"),'pt','-v7.3');
    fprintf('%.2f done\n',k/n);
end